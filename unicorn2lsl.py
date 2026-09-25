#!/usr/bin/env python3
"""
 Unicorn2lsl streams data from a Unicorn Hybrid Black EEG system to LSL

 Copyright (C) 2022 Robert Oostenveld
 GNU GPL v3+

 Modified by Adrien Bersia for the Intelligent User Interface and Human Computer Interaction courses taught at
 Maastricht University by Konstantia Zarkogianni

 STUDENTS: DO NOT MODIFY THIS FILE (except COM port)
"""

import serial
import struct
import string
import random
import numpy as np
from pylsl import StreamInfo, StreamOutlet

device = "COM8"
blocksize = 0.2
timeout = 5
nchan = 16
fsample = 250

start_acq = [0x61, 0x7C, 0x87]
stop_acq  = [0x63, 0x5C, 0xC5]

try:
    s = serial.Serial(device, 115200, timeout=timeout)
    print("connected to serial port " + device)
except:
    raise RuntimeError("cannot connect to serial port " + device)

lsl_name = 'Unicorn'
lsl_type = 'EEG'
lsl_format = 'float32'
lsl_id = ''.join(random.choice(string.digits) for i in range(6))

# create an outlet stream
info = StreamInfo(lsl_name, lsl_type, nchan, fsample, lsl_format, lsl_id)
outlet = StreamOutlet(info)

print('started LSL stream: name=%s, type=%s, id=%s' % (lsl_name, lsl_type, lsl_id))

# start the Unicorn data stream
s.write(bytes(start_acq))

response = s.read(3)
if response != b'\x00\x00\x00':
    raise RuntimeError("cannot start data stream")

print('started Unicorn')

try:
    while True:
        dat = np.zeros(nchan, dtype=np.float32)

        # read one block of data from the serial port
        payload = s.read(45)

        # check the start and end bytes
        if payload[0:2] != b'\xC0\x00':
            raise RuntimeError("invalid packet")
        if payload[43:45] != b'\x0D\x0A':
            raise RuntimeError("invalid packet")

        battery = 100 * float(payload[2] & 0x0F) / 15

        # ---- FIX 1: correct 24-bit sign handling for EEG ----
        eeg = np.zeros(8, dtype=np.float32)
        for ch in range(8):
            # read 24-bit big-endian as unsigned
            u24 = struct.unpack('>i', b'\x00' + payload[(3 + ch * 3):(6 + ch * 3)])[0]
            # if sign bit set, convert to signed by subtracting 2**24
            if (u24 & 0x00800000):
                u24 -= (1 << 24)
            eeg[ch] = float(u24) * 4500000.0 / 50331642.0

        accel = np.zeros(3, dtype=np.float32)
        # little-endian 16-bit signed
        accel[0] = float(struct.unpack('<h', payload[27:29])[0]) / 4096.0
        accel[1] = float(struct.unpack('<h', payload[29:31])[0]) / 4096.0
        accel[2] = float(struct.unpack('<h', payload[31:33])[0]) / 4096.0

        # ---- FIX 2: correct gyroscope byte indices to 33:38 ----
        gyro = np.zeros(3, dtype=np.float32)
        # little-endian 16-bit signed
        gyro[0] = float(struct.unpack('<h', payload[33:35])[0]) / 32.8
        gyro[1] = float(struct.unpack('<h', payload[35:37])[0]) / 32.8
        gyro[2] = float(struct.unpack('<h', payload[37:39])[0]) / 32.8

        counter = struct.unpack('<L', payload[39:43])[0]

        # collect the data that will be sent to LSL
        dat[0:8] = eeg
        dat[8:11] = accel
        dat[11:14] = gyro
        dat[14] = battery
        dat[15] = float(counter)

        # send the data to LSL
        outlet.push_sample(dat)

        if ((counter % fsample) == 0):
            print('received %d samples, battery %d %%' % (counter, battery))

except Exception as e:
    print('closing:', e)
    s.write(bytes(stop_acq))
    s.close()
    del outlet
