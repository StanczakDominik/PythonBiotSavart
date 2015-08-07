#!/bin/bash
echo $1
mkdir $1
mv RK4*x_positions.dat $1
mv RK4*y_positions.dat $1
mv RK4*z_positions.dat $1
mv RK4*energies* $1
cp RK4grid_B.dat $1
cp RK4grid_positions.dat $1
cp biot_params.py $1
