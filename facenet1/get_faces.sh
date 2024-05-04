#!/bin/sh
mkdir faces
i=500
while [ $i -ne 3333 ]
do
    i=$(($i+1))
    wget https://thispersondoesnotexist.com/ -O faces/$i.jpg
done