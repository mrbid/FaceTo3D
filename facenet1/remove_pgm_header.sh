mkdir faces3
for file in faces2/*
do
  if [ -f $file ]
  then
    tail +14c $file > faces3/$(basename "$file" .pgm).dat
  fi
done