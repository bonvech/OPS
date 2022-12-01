import pandas as pd
import struct


def read_channels(data):
    chan_len = 20 ## длина записи данных одного канала в байтах
    channels = []
    for i in range(15):
        ch = struct.unpack('dhhii', data[i * chan_len: (i + 1)* chan_len])
        #channels.append(ch)
        channels += ch
    return channels


def genetare_header():
    text_header = ['lenght', 'datatime', 'date', 'time', 'x1', 'x2', 'x3'] + \
                  ['y'+str(i) for i in range(1, 21)] + ['x4', 'temperature', 'pressure', 'humidity']
    chan_header = [x + str(i)  for i in range(1,16) for x in ["dt" , 'tr', 'al', 'vl', 'nch'] ]
    text_header += chan_header
    text_header += ['m' + str(i) for i in range(1, 6)] + ['TotalConc']
    text_header += ['z' + str(i) for i in range(1, 16)] + ['end']
    return text_header


def read_one_event():
    '''  Читаем одну запись  '''
    line_length = 658 ## длина одной строки данных
    chan_pos = 220    ## начало записи данных каналов в байтах
    chan_len = 300    ## длина записи данных всех каналов
    last_pos = 534    ## начало последних данных в записи, позиция первого байта после канало

    data_byte = file_handler.read(line_length)
    if len(data_byte) < line_length:
        print(f"\nError in data!!!! len = {len(data_byte)} bytes but expected {line_length} bytes" )
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
    channels = read_channels(data_byte[chan_pos:])

    ### read 7 integers after channels
    #print(chan_pos + chan_len + 5 * 2, last_pos)
    middle  = list(struct.unpack('5H', data_byte[chan_pos + chan_len: last_pos-4]))
    middle += list(struct.unpack('f',  data_byte[chan_pos + chan_len + 5 * 2: last_pos]))

    ### read last bytes
    last_num = struct.unpack('15dI', data_byte[last_pos:])

    alldata = list(firstdata) + list(channels) + list(middle) + list(last_num)
    return alldata



filename = "20+5 мин.54.O30"
text_header = genetare_header()
#print(text_header)

## читать бинарный файл
file_handler = open(filename, "rb")
# Читаем заголовок из файла
header_length = 9172
data_byte = file_handler.read(header_length)

### make dataframe
df = pd.DataFrame(columns=text_header)
while True:
    ## read one event
    alldata = read_one_event()
    if not alldata:
        break
    
    ### make dictionary
    newline = {x:y for x,y in zip(text_header, alldata)}
    
    df = pd.concat([df,pd.DataFrame([newline])], ignore_index=True)

    
#print(firstdata, channels, middle, last_num, sep='\n-----------------\n')
#print(alldata)
#print(last_num)
df['dt14']
df.to_csv(filename + ".txt", sep=' ') 
df.to_csv(filename + ".csv") 
