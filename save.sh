#!/bin/bash
echo $1
mkdir $1
mv *x_positions.dat $1
mv *y_positions.dat $1
mv *z_positions.dat $1
mv *energies* $1
cp grid_B.dat $1
cp grid_positions.dat $1
cp biot_params.py $1
