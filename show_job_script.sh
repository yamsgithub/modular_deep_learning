i=707767
until [ $i -gt 707784 ]
do
    #scontrol write batch_script $i
    echo $i
    sacct -o SubmitLine%200 -j $i
    ((i++))
done
