#!/usr/bin/octave -qf
pkg load signal

arg_list = argv ();
filename = arg_list{1};
b1 = csvread(filename);
[b,a] = butter(10, 20/(50/2));
csvwrite('matlab-filt20.csv', filter(b, a, b1));
[b,a] = butter(10, 15/(50/2));
csvwrite('matlab-filt15.csv', filter(b, a, b1));
[b,a] = butter(10, 10/(50/2));
csvwrite('matlab-filt10.csv', filter(b, a, b1));
[b,a] = butter(10, 6/(50/2));
csvwrite('matlab-filt6.csv', filter(b, a, b1));
[b,a] = butter(10, 2/(50/2));
csvwrite('matlab-filt2.csv', filter(b, a, b1));
