i=707767
until [ $i -gt  707784 ]
do
    echo $i
    scancel $i
    ((i++))
done       

