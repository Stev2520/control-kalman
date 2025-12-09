@echo off
g++ -static -std=c++17 -I"C:/msys64/ucrt64/include/eigen3" main.cpp CKF.cpp SRCF.cpp -o main