import os
import struct
import traceback
import pandas as pd
import numpy as np

from   datetime import datetime
import time

pd.set_option('display.max_columns', None)


## ----------------------------------------------------------------
##
## ----------------------------------------------------------------
def read_floats(data, n, size):
    return struct.unpack('d'*size, data[n:n + 8 * size])


## ----------------------------------------------------------------
##
## ----------------------------------------------------------------
def get_local_ip():
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return hostname, local_ip


## ----------------------------------------------------------------
##  extract year and month from datastring
## ----------------------------------------------------------------
def select_year_month(datastring):
    ## '2025-05-06 18:30:00'
    return '_'.join(datastring.split()[0].split("-")[:2])


############################################################################
############################################################################
class OPS:
    def __init__(self):
        self.debugmode = True #False
        self.device_name = "ops"
        self.model   = None
        self.sernum  = None
        self.sample_length = None

        ##  files and dirs
        self.sep = '/' if 'ix' in os.name else '\\' ## -- path separator for LINIX or Windows
        self.outdir = 'data'
        self.O30dirname = 'O30'  ## dir for raw data
        self.csvdirname = 'csv'  ## dir for data to site
        self.logdirname = "log"  ## dir for logs
        self.logfilename = f"{self.device_name}_log.txt"  ## file to write log messages
        self.dirlist = None
        self.extention = 'O30'   ## extention of raw file data
        self.curfilename = None
        self.prefix = None
        ## prepare dirs for data
        self.prepare_dirs()

        self.header = None
        self.dataheader = 'Sample #,Date,Start Time,Temp(C),Pressure(atm),Rel. Humidity,Errors,Alarm Triggered,Dilution Factor,Dead Time,Median,Mean,Geo. Mean,Mode,Geo. St. Dev.,Total Conc. '
        self.midheader = ['(µg/m³),Vol.Wt.Mean Diam.', '(µm²/cm³),Surf.Wt.Mean Diam.', '(#/cm³),Midpoint Diameter']
        self.weights = ['Mass', 'Surface', 'Number']
        self.units   = ['dW/dDp', 'dW/dlogDp', 'Concentration (dW)', '% Concentration', 'Raw Counts']
        self.boundaries = None
        self.LB = None
        self.UB = None


    ############################################################################
    ##  Print message to logfile
    ############################################################################
    def print_message(self, message, end=''):
        print(message)
        self.logfilename = self.logdirname + "_".join(["_".join(str(datetime.now()).split('-')[:2]),
                                                       self.device_name, 'log.txt'])
        with open(self.logfilename,'a') as flog:
            flog.write(str(datetime.now()) + ':  ')
            flog.write(message + end)


    ############################################################################
    ##  write message to bot
    ############################################################################
    def write_to_bot(self, text):
        try:
            hostname, local_ip = get_local_ip()
            text = f"{hostname} ({local_ip}): {self.ae_name}: {text}"

            bot = telebot.TeleBot(telebot_config.token, parse_mode=None)
            bot.send_message(telebot_config.channel, text)
            self.print_message(f"Sent to bot: {text}", '\n')  ## write to log file
        except Exception as err:
            ##  напечатать строку ошибки
            text = f": ERROR in writing to bot: {err}"
            self.print_message(text, '\n')  ## write to log file


    ############################################################################
    ##  check and create dirs for data
    ############################################################################
    def prepare_dirs(self):
        if self.outdir[-1] != self.sep:
            self.outdir = self.outdir + self.sep
        if not os.path.isdir(self.outdir):
            os.makedirs(self.outdir)

        ##  check and create dirs for data
        for dirname in [self.O30dirname, self.csvdirname, self.logdirname]: #, 'table']: ##'raw'
            path = f"{self.outdir}{dirname}{self.sep}"
            if not os.path.isdir(path):
                os.makedirs(path)
            ##  rename dirs
            if dirname == self.logdirname:
                self.logdirname = path
            if dirname == self.csvdirname:
                self.csvdirname = path
            if dirname == self.O30dirname:
                self.O30dirname = path
        #print([self.O30dirname, self.csvdirname, self.logdirname, self.sep])


    ############################################################################
    ## Return sorted list of data files
    ############################################################################
    def get_dirlist(self, dirname):
        dirlist = [x for x in os.listdir(dirname) if x.split('.')[-1] == 'O30']

        ## сравнить префиксы у всех файлов O30
        ## префиксом считается часть имени файла до первой точки
        prefix = set(x.split('.')[0] for x in dirlist)
        #print(len(prefix), prefix)
        if len(prefix) == 1:
            self.prefix = prefix.pop()
        elif len(prefix) > 1:
            print(f"В папке  {dirname} есть данные разных серий. Положите каждую серию в свою папку и запустите программу для каждой серии отдельно.")
            return []
        else:
            print(f"В папке {dirname} нет файлов с данными. Положите данные в папку и запустите программу снова.")
            return []

        ## отсортировать в порядке возрастания номеров
        dirlist = sorted(dirlist,
                        key=lambda x: int(x.split('.')[-2]) if x.split('.')[1:-1] else 0)
        return dirlist


    ############################################################################
    ##  Get last file name
    ##  найти самый поздний файл
    ############################################################################
    def get_latest_file(self, dirname):
        max_file = 'Initial string'

        sep = self.sep
        if not dirname.endswith(self.sep):   dirname += sep
        if not os.path.isdir(dirname):
            print(f"Alarm!! Нет такой папки {dirname}! Валим отсюда!")
            return "Error" ## \todo вернуть сообщение об ошибке

        max_atime = 0
        if self.debugmode:
            print(os.listdir(dirname))
        for filename in os.listdir(dirname):
            ## проверить файл ли это
            if not os.path.isfile(dirname + filename):
                continue
            if not filename.endswith(self.extention):
                continue
            if os.path.getmtime(dirname + filename) > max_atime:
                max_atime = os.path.getmtime(dirname + filename)
                max_file = filename
        return max_file


    ############################################################################
    ## translate header binary file
    ############################################################################
    def translate_header(self, header):
        model_byte = 64
        sernum_byte = model_byte + 20
        model  = header[model_byte:model_byte + 4].decode()
        sernum = header[sernum_byte:sernum_byte + 10].decode()
        sample_length_byte = 2170 * 4
        sample_length = struct.unpack('i', header[sample_length_byte:sample_length_byte + 4])[0]
        boundaries = read_floats(header, 8692, 17) ## Boundaries

        ## \todo сравнение заголовка текущего файла с заголовком предыдущего

        ## header 1
        header1 = '\n'.join([
                        f"Instrument Model,{model}",
                        f'Instrument Serial Number,{sernum}',
                        f'Sample Length (s),{sample_length}',
                        'Alarm Set,No',
                        'Dead Time Correction Applied,Yes'
                        ])

        header2 = '\n'.join([
                        f'Refractive Index Applied,No', '',
                        ',,,,,,,,,,,,,,,,LB,' + ','.join(map(str,boundaries[:-1])),
                        ',,,,,,,,,,,,,,,,UB,' + ','.join(map(str,boundaries[1:])),
                        ',,,,,,,,,,,,,,,,LB with RI,' + ','.join(map(str,boundaries[:-1])) + ',',
                        ',,,,,,,,,,,,,,,,UB with RI,' + ','.join(map(str,boundaries[1:])) + ','
                        ])

        self.header = [header1, header2]
        self.boundaries = boundaries[:]
        self.LB = boundaries[:-1]
        self.UB = boundaries[1:]
        self.dB = [(u - l) for l, u in zip(self.LB, self.UB)]
        self.model = model
        self.sernum = sernum
        self.sample_length = sample_length


    ############################################################################
    ############################################################################
    def print_header(self, unit, weight):
        header = ''
        ## check unit and weight
        if unit > len(self.units) - 1 or weight > len(self.weights) - 1:
            print(f"Out of parameter range. Unexpected error in function {traceback.extract_stack()[-1][2]}")
            return 1
        ## Для Raw считаем только Number
        if unit == len(self.units) - 1:
            weight = len(self.weights) - 1

        #print(f"Sample file, {self.curfilename}")
        #print(self.header[0])
        header += f"Sample file, {self.curfilename}\n"
        header += (self.header[0] + '\n')

        ## units
        #print(f'Units,{self.units[unit]}')
        header += f'Units,{self.units[unit]}\n'

        ## parameter
        #print(f'Weight,{self.weights[weight]}')
        #print(self.header[1])
        header += f'Weight,{self.weights[weight]}\n'
        header += self.header[1] + '\n'

        return header


    ############################################################################
    ## читать из данных события значения в каналах
    ############################################################################
    def read_channels(self, data):
        chan_len = 20 ## длина записи данных одного канала в байтах
        channels = []
        for i in range(15):
            ch = struct.unpack('dhhii', data[i * chan_len: (i + 1)* chan_len])
            #channels.append(ch)
            channels += ch
        return channels


    ############################################################################
    ############################################################################
    def generate_csv_header(self):
        text_header = ['lenght', 'datetime', 'Date', 'Start Time', 'x1', 'x2', 'x3'] + \
                    ['y'+str(i) for i in range(1, 21)] + ['x4', 'Temp(C)', 'Pressure(atm)', 'Rel. Humidity']
        chan_header = [x + str(i)  for i in range(1,16) for x in ["dt" , 'tr', 'al', 'vl', 'nch'] ]
        text_header += chan_header
        text_header += ['m' + str(i) for i in range(1, 4)] # + ['TotalConc']
        text_header += ['z' + str(i) for i in range(0, 16)] + ['other']
        #text_header += [f"{(self.LB[i] + self.UB[i]) * 0.5:.3f}" for i in range(0, 16)] + ['other']
        ## tr14 - Errors:
        text_header[text_header.index('tr14')] = "Errors"
        ## dt14 - Dead Time - заменить в заголовке
        text_header[text_header.index('dt14')] = "Dead Time"
        return text_header


    ############################################################################
    ############################################################################
    def read_one_event(self, file_with_data):
        '''  Читаем одну запись  '''
        line_length = 658 ## длина одной строки данных
        chan_pos = 220    ## начало записи данных каналов в байтах
        chan_len = 300    ## длина записи данных всех каналов
        last_pos = 534 - 8    ## начало последних данных в записи, позиция первого байта после канало

        ## read binary data from file
        data_byte = file_with_data.read(line_length)

        ## check length of read line
        if len(data_byte) == 0:
            return

        if len(data_byte) < line_length:
            print(f"\nError in data!!!! len = {len(data_byte)} bytes but expected {line_length} bytes" )
            return

        ## check first two chars: if no 92 02 - no data
        if not data_byte.startswith(b'\x92' + b'\x02'):
            print(f"\nError in data!! No standart start bytes!!! No standart length of event data") ## \todo правильно обработать эту ошибку
            return

        ### read first bytes
        format_first = 'ii6hiiI' + 23 * 'd' +'f'
        firstdata = list(struct.unpack(format_first, data_byte[:chan_pos]))
        ## соберем дату
        firstdata.insert(2, str(firstdata[4]) + '-' + str(f'{firstdata[2]:02d}') + '-' + str(f'{firstdata[3]:02d}'))
        [firstdata.pop(3) for _ in range(3)]
        ## соберем время
        firstdata.insert(3, str(f'{firstdata[3]:02d}') + ':' + str(f'{firstdata[4]:02d}') + ':' + str(f'{firstdata[5]:02d}'))
        [firstdata.pop(4) for _ in range(3)]

        ### read channels
        channels = self.read_channels(data_byte[chan_pos:])

        ### read 7 integers after channels
        #print(chan_pos + chan_len + 5 * 2, last_pos)
        middle  = list(struct.unpack('3H', data_byte[chan_pos + chan_len: last_pos]))
        #middle += list(struct.unpack('f',  data_byte[chan_pos + chan_len + 5 * 2: last_pos]))

        ### read last bytes
        last_num = struct.unpack('16dI', data_byte[last_pos:])

        alldata = list(firstdata) + list(channels) + list(middle) + list(last_num)
        return alldata


    ############################################################################
    ############################################################################
    def read_binary_data(self, file_with_data):
        ### make dataframe
        text_header = self.generate_csv_header()
        df = pd.DataFrame(columns=text_header)

        ## Читаем файл, пока считываются строки по 658 символов \todo заменить число на переменную
        event_length = 658
        n = 0
        reading = True
        while True:
            ## read one event
            alldata = self.read_one_event(file_with_data)
            if not alldata:
                break

            ### make dictionary
            newline = {x:y for x,y in zip(text_header, alldata)}
            if newline:
                #print(f"newline: {newline}")
                if df.shape[0] == 0:
                    df = pd.DataFrame([newline])
                else:
                    df = pd.concat([df, pd.DataFrame([newline])], ignore_index=True)

        #print(firstdata, channels, middle, last_num, sep='\n-----------------\n')
        #print(alldata)
        #print(last_num)
        #df['dt14']
        #print(df[['date', 'time', 'Dead Time']])

        ##  change error code to text Errors:
        errors = { 0: "No Errors",
                  48: 'Flow Error;Flow Blocked',
                 264: 'System Error;Flow Blocked Instrument Stopped'
                 }

        unknown_errors = [x for x in set([0, 48, 264]) if not x in errors.keys()]
        if len(unknown_errors):
            ## write message to bot \todo
            print(f"There are new unknown errors in errors.keys(): {unknown_errors}")

        ##  change error code to text
        df['ErrorCode'] = df['Errors']
        df['Errors'] = df['Errors'].apply(lambda x: 'No Errors'               if x == 0 else
                                                    'Flow Error;Flow Blocked' if x == 48 else
                                                    'System Error;Flow Blocked Instrument Stopped' if x == 264 else
                                                    f'unknown error: {x}')

        df['Pressure(atm)'] = df['Pressure(atm)'] / 101.33
        #df.to_csv(self.outpath + self.curfilename + "raw.txt", sep=' ')
        #df.to_csv(self.outpath + self.curfilename + "raw.csv")

        return df


    ############################################################################
    ## get name of output file
    ############################################################################
    def get_output_filename(self, weight, unit):
        ## --------------------
        ## Create filename
        #self.weights = ['Mass', 'Surface', 'Number']
        #self.units   = ['dW/dDp', 'dW/dlogDp', 'Concentration (dW)', '% Concentration', 'Raw Counts']
        nameunit = ["dW_dDp", "dW_dlogDp", "dW", "%", "raw"]
        filename = self.prefix + '_' + self.weights[weight].lower()[:4] + '_' + nameunit[unit] + '.csv'
        return filename


    ############################################################################
    ## create data header of output scv file
    ############################################################################
    def get_output_csv_header(self, weight, unit):
        ## --------------------
        ## Create file header
        header = 'Sample #,Date,Start Time,Temp(C),Pressure(atm),Rel. Humidity,Errors,Alarm Triggered,Dilution Factor,Dead Time,Median,Mean,Geo. Mean,Mode,Geo. St. Dev.,Total Conc. '
        header += self.midheader[weight] + ','

        ## add middle diameters
        if self.weights[weight] == 'Mass':
            header +=  ','.join([f'{x:.3f}' for x in self.Dpv[:]]) + ',,'
        elif self.weights[weight] == 'Surface':
            header +=  ','.join([f'{x:.3f}' for x in self.Dps[:]]) + ',,'
        elif self.weights[weight] == 'Number':
            header +=  ','.join([f'{x:.3f}' for x in self.Dp[:]]) + ',,'
        else:
            print("Error! Impossible argument!")
            return 1

        return header


    ############################################################################
    ## translate all data files from directory in path
    ##  last=1 - only last file, last=0 - all files
    ############################################################################
    def translate_data(self, path, last=0):
        self.dirlist = self.get_dirlist(path)
        # если нет файлов - прекратить
        if not self.dirlist:
            return 1

        ##  create output directory
        outpath = path + 'out/'
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        self.outpath = outpath

        ## перебрать один или все файлы
        files = [self.get_latest_file(path)] if last else self.dirlist
        for filename in files:
            print("===================")
            self.curfilename = filename
            print(f"File: {filename}")
            ## !!!! \todo сравнить шапки файлов. Если шапки разные, то обрабатывать только по шапке первого файла.
            ## или по двум шапкам ?????

            self.translate_ops_file(path + filename)


    ############################################################################
    ##  read one 'O30' file
    ############################################################################
    def read_ops_file(self, filename):
        ##  read binary file
        with open(filename, "rb") as file_handler:
            ## read header
            header = file_handler.read(9172)
            header = self.translate_header(header)
            #self.print_header(4, 2)

            ## read binary data
            data = self.read_binary_data(file_handler)
            #print(data)

        ##  particle diameter (channel midpoint)
        LB = pd.Series(self.LB)
        UB = pd.Series(self.UB)
        ## particle diameter (channel midpoint)
        self.Dp = (LB + UB) / 2
        self.dDp = UB - LB
        self.Dps = LB * ((1 + (UB/LB) + (UB/LB)**2)/3) ** 0.5
        self.Dpv = LB * ((1 + (UB/LB)**2) * (1 + (UB/LB)) / 4) ** (1/3)

        ## dW/dDp size distribution to log channel width
        self.dlogDp = np.log10(UB) - np.log10(LB)

        return data


    ############################################################################
    ##  translate one 'O30' file
    ############################################################################
    def translate_ops_file(self, filename):
        self.curfilename = filename
        ##  read one 'O30' file
        data = self.read_ops_file(filename)

        ## calculate all parameters
        data = self.calculate_all(data)

        ###################################
        ## calculate number log
        #self.weights = ['Mass', 'Surface', 'Number']
        #self.units   = ['dW/dDp', 'dW/dlogDp', 'Concentration (dW)', '% Concentration', 'Raw Counts']
        for weight in [0, 2]: #range(len(self.weights)): ##  ['Mass', 'Surface', 'Number']
            for unit in [0, 1, 2, 3, 4]: #range(len(self.units)): ##  ['dW/dDp', 'dW/dlogDp', 'Concentration (dW)', '% Concentration', 'Raw Counts']
                ## Для 'Mass', 'Surface' считаем только четыре первых юнита
                if weight < len(self.weights) - 1:
                    if unit == len(self.units) - 1:
                        continue

                #self.print_header(unit, weight)
                print(f"weight = {self.weights[weight]:7s}, unit = {self.units[unit]:16s}", end='\t')
                #print(self.get_output_filename(weight, unit))
                #print(self.get_putput_csv_header(weight, unit))
                self.print_data_to_output_file(data, weight, unit)


    ############################################################################
    ############################################################################
    def print_data_to_output_file(self, data, weight, unit):
        data = data.copy()
        ## имя веса: ['Mass', 'Surface', 'Number']
        weightname = self.weights[weight]
        unitname   = self.units[unit]

        ## get filename and columns for output dataset
        filename = self.get_output_filename(weight, unit)
        #print('\n', " =="*10, '\n', weightname, unitname, filename)
        print("file: ", self.outpath + filename)
        filename = self.outpath + filename

        datacolumns = self.get_output_csv_header(weight, unit).split(',')
        #print("File header:")
        #self.print_header(unit, weight)
        #print("CSV Header:", self.get_output_csv_header(weight, unit), sep='\n')

        ## rename columns
        out = ['Total Conc. (µg/m³)', 'Vol.Wt.Mean Diam.', 'Mode','Mean','Geo. Mean', 'Geo. St. Dev.']
        if weightname == "Mass":
            ext = ['M','null','moda_m','mean_m','gmean_m','gsigma_m']
        elif weightname == "Number":
            out[0:2] = ['Total Conc. (#/cm³)', 'Midpoint Diameter']
            ext = ['N','null','moda_n','mean_n','gmean_n','gsigma_n']
        elif weightname == "Surface":
            out[0:2] = ['Total Conc. (µm²/cm³)', 'Surf.Wt.Mean Diam.']
            ext = ['S','null','moda_s','mean_s','gmean_s','gsigma_s']

        ## add middle diameters
        ## ['dW/dDp', 'dW/dlogDp', 'Concentration (dW)', '% Concentration', 'Raw Counts']
        if weightname == 'Mass':
            out += [f'{x:.3f}' for x in self.Dpv[:]]
        elif weightname == 'Surface':
            out += [f'{x:.3f}' for x in self.Dps[:]]
        elif weightname == 'Number':
            out += [f'{x:.3f}' for x in self.Dp[:]]
        else:
            print("Error! Impossible argument!")
            return 1

        if unitname == 'dW/dDp':
            ext += [weightname.lower()[0] + 'dDp'    + str(i) for i in range(0, 16)]
        elif unitname == 'dW/dlogDp':
            ext += [weightname.lower()[0] + 'dlogDp' + str(i) for i in range(0, 16)]
        elif unitname == 'Concentration (dW)':
            ext += [weightname.lower()[0]            + str(i) for i in range(0, 16)]
        elif unitname == '% Concentration':
            ext += [weightname.lower()[0] + 'per'    + str(i) for i in range(0, 16)]
        elif unitname == 'Raw Counts':
            ext += [                        'z'      + str(i) for i in range(0, 16)]

        data['null'] = ''
        data[''] = ''
        data['Sample #'] = data.index

        ##  переименовать имеющиеся столбцы
        newcolumns = dict(zip(out, ext))
        for key in newcolumns:
            if key not in data.columns:
                data[key] = data[newcolumns[key]]

        ## lenght datetime Date Start Time x1 x2 x3
        ## y1 y2 y3 y4 y5 y6 y7 y8 y9 y10 y11 y12 y13 y14 y15 y16 y17 y18 y19 y20 x4
        ## Temp(C) Pressure(atm) Rel. Humidity dt1 tr1 al1 vl1 nch1 dt2 tr2 al2 vl2 nch2
        ## dt3 tr3 al3 vl3 nch3 dt4 tr4 al4 vl4 nch4 dt5 tr5 al5 vl5 nch5 dt6 tr6 al6 vl6
        ## nch6 dt7 tr7 al7 vl7 nch7 dt8 tr8 al8 vl8 nch8 dt9 tr9 al9 vl9 nch9
        ## dt10 tr10 al10 vl10 nch10 dt11 tr11 al11 vl11 nch11 dt12 tr12 al12 vl12 nch12 dt13
        ## tr13 al13 vl13 nch13 Dead Time Errors al14 vl14 nch14 dt15 tr15 al15 vl15 nch15
        ## m1 m2 m3 z0 z1 z2 z3 z4 z5 z6 z7 z8 z9 z10 z11 z12 z13 z14 z15 other
        ## Alarm Triggered Dilution Factor Median kk n0 n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 n15
        ## ndDp0 ndDp1 ndDp2 ndDp3 ndDp4 ndDp5 ndDp6 ndDp7 ndDp8 ndDp9 ndDp10 ndDp11 ndDp12 ndDp13 ndDp14 ndDp15
        ## ndlogDp0 ndlogDp1 ndlogDp2 ndlogDp3 ndlogDp4 ndlogDp5 ndlogDp6 ndlogDp7 ndlogDp8 ndlogDp9 ndlogDp10 ndlogDp11 ndlogDp12 ndlogDp13 ndlogDp14 ndlogDp15
        ## N nper0 nper1 nper2 nper3 nper4 nper5 nper6 nper7 nper8 nper9 nper10 nper11 nper12 nper13 nper14 nper15
        ## moda_n mean_n gmean_n gsigma_n v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15
        ## m0 m4 m5 m6 m7 m8 m9 m10 m11 m12 m13 m14 m15
        ## mdDp0 mdDp1 mdDp2 mdDp3 mdDp4 mdDp5 mdDp6 mdDp7 mdDp8 mdDp9 mdDp10 mdDp11 mdDp12 mdDp13 mdDp14 mdDp15
        ## mdlogDp0 mdlogDp1 mdlogDp2 mdlogDp3 mdlogDp4 mdlogDp5 mdlogDp6 mdlogDp7 mdlogDp8 mdlogDp9 mdlogDp10 mdlogDp11 mdlogDp12 mdlogDp13 mdlogDp14 mdlogDp15
        ## M mper0 mper1 mper2 mper3 mper4 mper5 mper6 mper7 mper8 mper9 mper10 mper11 mper12 mper13 mper14 mper15
        ## moda_m mean_m gmean_m gsigma_m null
        ## Sample # Total Conc. (µg/m³) Vol.Wt.Mean Diam. Mode Mean Geo. Mean Geo. St. Dev.
        ## 0.374 0.502 0.687 0.904 1.103 1.303 1.502 1.702 1.902 2.102 2.370 2.747 3.213 4.614 6.888 9.052
        ## Total Conc. (#/cm³) Midpoint Diameter
        ## 0.370 0.500 0.680 0.900 1.100 1.300 1.500 1.700 1.900 2.100 2.367 2.742 3.207 4.531 6.816 9.016

        ##  write to file
        outdata = data[datacolumns]
        with open(filename, 'w') as outfile:
            outfile.write(self.print_header(unit, weight))

        ##  add table data
        outdata.to_csv(filename, index=False, mode='a', float_format='%.5f', date_format='%m/%d/%Y')


    ############################################################################
    ############################################################################
    def calculate_all(self, data):
        data = data.copy()
        ## global ndDp
        Q  = 16.666666666  ## расход пробы (см3/сек)  ## sample flow rate
        fi = 1

        tz = self.sample_length
        td = data['Dead Time']

        data['Alarm Triggered'] = 'NA'
        data['Dilution Factor'] = 1
        data['Median'] = 0
        #print(data.head(2))

        commoncolumns = ['Date', 'Start Time', 'Temp(C)', 'Pressure(atm)', 'Rel. Humidity', 'Errors', 'Alarm Triggered', 'Dilution Factor', 'Dead Time', 'Median']
        #print(data[commoncolumns].head(2))

        ###########################################################
        ##  Концентрация Concentration
        ##  number weighted concentration per channel -> 16 numbers
        data['kk'] = fi / (Q * (tz-td))  ## coefficient
        for i in range(0, 16):
            data['n' + str(i)] = data['z' + str(i)] * data['kk']
        #print(data['kk'])

        ##  изменить тип данных z на целочисленный
        data = data.astype({'z' + str(i): np.int64 for i in range(16)})

        ##!!! n = data[['n' + str(i) for i in range(0, 16)]]
        #print("\n\n\n !!!n:\n", n)

        ##  Для разных weight нужно писать переменные с одинаковыми названиями, но по своим формулам
        ##  и в зависимости от своих units

        ##  dW/dDp size distribution to channel width in um
        #for i in range(0, 16):
        #    data['ndDp' + str(i)] = data['n' + str(i)].copy() / self.dDp[i]
        #ndDp = data[['ndDp' + str(i) for i in range(0, 16)]]
        ndDp = {}
        for i in range(0, 16):
            ndDp['ndDp' + str(i)] = data['n' + str(i)].copy() / self.dDp[i]
        ndDp = pd.DataFrame(ndDp)
        data = pd.concat([data, ndDp], axis=1)
        if self.debugmode:
            print(f"ndDp: {ndDp}")

        ##  dW/dDp size distribution to log channel width
        ##  ndlogDp = n / dlogDp
        #for i in range(0, 16):
        #    data['ndlogDp' + str(i)] = data['n' + str(i)].copy() / self.dlogDp[i]
        #ndlogDp = data[['ndlogDp' + str(i) for i in range(0, 16)]]
        ndlogDp = {}
        for i in range(0, 16):
            ndlogDp['ndlogDp' + str(i)] = data['n' + str(i)].copy() / self.dlogDp[i]
        ndlogDp = pd.DataFrame(ndlogDp)
        data = pd.concat([data, ndlogDp], axis=1)

        ## ------------------------------
        ## 'Number'
        ## ------------------------------
        ##  Total number concentration
        ##  N = sum(n)
        data['N'] = np.sum(data[['n' + str(i) for i in range(0, 16)]], axis=1)

        ##  Percent concentration
        ##  nper = n * 100 / N
        for i in range(0, 16):
            data['nper' + str(i)] = data['n' + str(i)] * 100 / data['N']
        #nper = data[['nper' + str(i) for i in range(0, 16)]]
        #print("nper:\n", nper)

        ##  Mode
        ##  moda_n = Dp[ndDp.idxmax()]
        data['moda_n'] = ndDp.apply(lambda row: self.Dp[list(ndDp.columns).index(row.idxmax())], axis=1)
        #print(data['moda_n'])
        #moda_n = self.Dp[ndDp.idxmax(axis=1).values]

        ##  Mean
        ##  mean_n = sum(n * Dp) / N
        arr = data.loc[:,['n' + str(i) for i in range(0, 16)]]
        for i in range(len(arr.columns)):
            arr['n' + str(i)] = arr['n' + str(i)] * self.Dp[i]
        data['mean_n'] = np.sum(arr, axis=1) / data['N']

        ##  Geometric Mean
        ##  gmean_n = np.exp(sum(n * np.log(Dp)) / N)
        arr = data.loc[:,['n' + str(i) for i in range(0, 16)]]
        for i in range(len(arr.columns)):
            arr['n' + str(i)] = arr['n' + str(i)] * np.log(self.Dp[i])
        data['gmean_n'] = np.exp(np.sum(arr, axis=1) / data['N'])
        #print(data['gmean_n'])

        ##  Geometric Standart Deviation
        ##  gsigma_n = np.exp((sum(n * (np.log(Dp) - np.log(gmean_n)) ** 2) / N) ** 0.5)
        arr = data.loc[:,['n' + str(i) for i in range(0, 16)]]
        for i in range(len(arr.columns)):
            arr['n' + str(i)] = arr['n' + str(i)] * (np.log(self.Dp[i]) - np.log(data['gmean_n'])) ** 2
        data['gsigma_n'] = np.exp((np.sum(arr, axis=1) / data['N']) ** 0.5)
        #print(data[['Date', 'Start Time', 'N','moda_n', 'mean_n', 'gmean_n', 'gsigma_n']])


        ## ------------------------------
        ## Масса
        ## ------------------------------
        ##  v = np.pi * self.Dpv ** 3 * n / 6
        for i in range(0, 16):
            data['v' + str(i)] = data['n' + str(i)] * (self.Dpv[i] ** 3) * np.pi / 6
        v = data[['v' + str(i) for i in range(0, 16)]]
        #print("v:", v)

        ##  mass weighted concentration per channel
        ##  m = ro * v
        ro = 1  ## particle density
        for i in range(0, 16):
            data['m' + str(i)] = data['v' + str(i)] # * ro
        m = data[['m' + str(i) for i in range(0, 16)]]
        #print("m:")
        #print(data[['Date', 'Start Time'] + ['m' + str(i) for i in range(0, 6)] + ['m' + str(i) for i in range(13, 16)]])

        ##  dW/dDp mass distribution to channel width in um
        ##  mdDp = m / dDp
        #for i in range(0, 16):
        #    data['mdDp' + str(i)] = data['m' + str(i)] / self.dDp[i]
        #mdDp = data[['mdDp' + str(i) for i in range(0, 16)]]
        mdDp = {}
        for i in range(0, 16):
            mdDp['mdDp' + str(i)] = data['m' + str(i)] / self.dDp[i]
        mdDp = pd.DataFrame(mdDp)
        data = pd.concat([data, mdDp], axis=1)
        #print("mdDp:\n", mdDp)
        #print("Mediana:\n", mdDp.median(axis=1))

        ##  dW/dDp size distribution to log channel width
        ##  mdlogDp = m / dlogDp
        ##  old:
        #for i in range(0, 16):
            #data['mdlogDp' + str(i)] = data['m' + str(i)] / self.dlogDp[i]
        #mdlogDp = data[['mdlogDp' + str(i) for i in range(0, 16)]]
        mdlogDp = {}
        for i in range(0, 16):
            mdlogDp['mdlogDp' + str(i)] = data['m' + str(i)] / self.dlogDp[i]
        mdlogDp = pd.DataFrame(mdlogDp)
        data = pd.concat([data, mdlogDp], axis=1)
        #print("mdlogDp:\n", mdlogDp)

        ### Total mass concentration
        # M = sum(m)
        data['M'] = np.sum(m, axis=1)
        #print("M:", data['M'])

        ## Percent concentration
        #mper = m * 100 / M
        for i in range(0, 16):
            data['mper' + str(i)] = data['m' + str(i)] * 100 / data['M']
        mper = data[['mper' + str(i) for i in range(0, 16)]]
        #print("mper:\n", mper)

        ## ------------------------------
        # Mode
        # moda_m = Dp[mdDp.idxmax()]
        data['moda_m'] = mdDp.apply(lambda row: self.Dpv[list(mdDp.columns).index(row.idxmax())], axis=1)
        #print("data['moda_n']", data['moda_n'])
        #moda_n = self.Dp[ndDp.idxmax(axis=1).values]

        ## Mean
        #mean_m = sum(m * Dpv) / M
        arr = data.loc[:,['m' + str(i) for i in range(0, 16)]]
        for i in range(len(arr.columns)):
            arr['m' + str(i)] = arr['m' + str(i)] * self.Dpv[i]
        data['mean_m'] = np.sum(arr, axis=1) / data['M']

        ## Geometric Mean
        ## gmean_n = np.exp(sum(n * np.log(Dp)) / N)
        arr = data.loc[:,['m' + str(i) for i in range(0, 16)]]
        for i in range(len(arr.columns)):
            arr['m' + str(i)] = arr['m' + str(i)] * np.log(self.Dpv[i])
        data['gmean_m'] = np.exp(np.sum(arr, axis=1) / data['M'])

        ## Geometric Standart Deviation
        ## gsigma_n = np.exp((sum(n * (np.log(Dp) - np.log(gmean_n)) ** 2) / N) ** 0.5)
        arr = data.loc[:,['m' + str(i) for i in range(0, 16)]]
        for i in range(len(arr.columns)):
            arr['m' + str(i)] = arr['m' + str(i)] * (np.log(self.Dpv[i]) - np.log(data['gmean_m'])) ** 2
        data['gsigma_m'] = np.exp((np.sum(arr, axis=1) / data['M']) ** 0.5)
        #print(data[['Date', 'Start Time', 'M', 'moda_m', 'mean_m', 'gmean_m', 'gsigma_m']])

        return data


    ############################################################################
    ##  translate one 'O30' file to site format
    ############################################################################
    def translate_ops_file_to_site(self, filename):
        self.curfilename = filename
        ##  read one 'O30' file
        data = self.read_ops_file(filename)

        ##  calculate all parameters
        data = self.calculate_all(data)

        ##  rename columns
        columns = {'datetime':"timestamp", 'M': 'Total Conc. (µg/m³)'}
        data = data.rename(columns=columns, errors="raise")

        data['datetime'] = data.apply(lambda row: datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S'), axis=1)

        ##  select data to write
        columns = ['timestamp', 'datetime',  'Total Conc. (µg/m³)', 'Pressure(atm)', # 'Date', 'Start Time',
                   'ErrorCode']
        data = data[columns]
        if self.debugmode:
            print(data.columns, data)

        ##  save result to files yyyy_mm_ops.csv
        self.print_data_for_site_to_files(data)

        ##  check errors
        ##


    ## ----------------------------------------------------------------
    ##  save one file
    ## ----------------------------------------------------------------
    def add_data_to_csv_files(self, dfsave, ym_pattern):
        filename = f"{self.csvdirname}{ym_pattern}_{self.device_name}.csv"

        ## create new file
        if not os.path.exists(filename):
            #dfsave.to_csv(filename, index=False)
            text = f"New {filename} created"
            self.print_message(text, '\n')
            #self.write_to_bot(text)
        else:
            ## read existing csv file
            dfexist = pd.read_csv(filename)
            dfsave = pd.concat([dfexist, dfsave], ignore_index=True)\
                                .drop_duplicates(subset=['timestamp'])\
                                .sort_values(by=['timestamp'])
            dfsave['Pressure(atm)'] = dfsave['Pressure(atm)'].astype(float)

        ##  write to file
        dfsave.to_csv(filename, index=False, float_format='%g') #'%.5f',)


    ## ----------------------------------------------------------------
    ##  save data to files for site
    ## ----------------------------------------------------------------
    def print_data_for_site_to_files(self, data):
        ##
        if not data.empty:
            if self.debugmode:
                print(data.head(1))
        else:
            print_message("Empty data to write to files!")
            print("exit...")
            return

        ##  extract year and month from data
        year_month = data['datetime'].apply(select_year_month).unique()
        if self.debugmode:
            print(f"year_month: {year_month}")

        ##  write to table files
        for ym_pattern in year_month:
            dfsave = data[data['datetime'].apply(select_year_month) == ym_pattern]
            
            text = f"{ym_pattern}: {dfsave.shape}"
            self.print_message(text, '\n')

            if not dfsave.empty:
                self.add_data_to_csv_files(dfsave, ym_pattern)


    ##  ----------------------------------------------------------------
    ##  Parse status errors
    ##  ----------------------------------------------------------------
    def parse_errors(errors):
        known_errors = { 0: "No Errors",
                        # 7, 34
                        48: 'Flow Error;Flow Blocked',
                       264: 'System Error;Flow Blocked Instrument Stopped'
                        }

        errors = set(errors)
        errors.remove(0)

        if len(errors) == 0:
            return ""

        unknown_errors = [x for x in errors if not x in known_errors.keys()]
        if len(unknown_errors):
            ## write message to bot \todo
            self.print_message(f"There are new unknown errors: {unknown_errors}")

        message = ""
        for error in errors:
            message += f"{error}: {known_errors[error]} \n"
        return message


## --------------------------------------------------------------------------------------------------
## --------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ======================================
    dirname =   "yadisk_data/" # "data/2025-03-16/" #"data/2023-09-19/" #"Satino/" #

    ## \todo сделать имя папки входным параметром, чтоб можно было указать любую папку и получить полный путь к ней

    # ======================================
    ops = OPS()
    pwd = os.getcwd() + ops.sep
    dirname = dirname if dirname.endswith(ops.sep) else f"{dirname}{ops.sep}"
    print(pwd + dirname)

    # ======================================
    filename = dirname + ops.get_latest_file(dirname)
    print(f"filename: {filename}")
    filename = "yadisk_data/2025-05-06_19+1.1.O30"
    filename = "yadisk_data/old/2025-05-06_19+1.O30.old"
    filename = "yadisk_data/2025-05-06_19+1.O30"
    filename = "yadisk_data/Chashn_2025-05-22_19+1.1.O30"
    filename = "yadisk_data/Chashn_2025-05-22_19+1.1.1.O30"
    ops.translate_ops_file_to_site(filename)

    #ops.translate_data(pwd + dirname + ops.sep, last=0) ## last=1 один последний файл

## проверили логарифм массы и числа
## осталось проверить логарифм surface \todo
