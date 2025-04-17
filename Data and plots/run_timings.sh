#The csv formatted data gets output to stderr
#so this should be run like bash timings.sh 2>timings.csv

ITERS=1000

echo 1>&2 "height,width,iterations,precision,skip_cpu,device_index,block_size,cpu_iteration,gpu_iteration,gpu_iteration_setup,gpu_iteration_alloc,gpu_iteration_to,gpu_iteration_comp,gpu_iteration_from,iteratation_diff,iteration_disc,cpu_average,gpu_average,gpu_average_setup,gpu_average_alloc,gpu_average_to,gpu_average_comp,gpu_average_from,average_diff,average_disc,cpu_average_val,gpu_average_val"

#CPU timings

SIZES=(960 1920 3840 7680 15360)
for size in "${SIZES[@]}"
do
    for i in {1..3}
    do
        ./assignment2 -n $size -m $size -b 256 -d 0 -p $ITERS -a -o
    done
done

#Small block size timings

SIZES=(960 1920 3840 7680 15360)
BLOCKS=(4 8 16)
DEVICES=(0)
for size in "${SIZES[@]}"
do
    for block_size in "${BLOCKS[@]}"
    do
        for i in {1..3}
        do
            for device in "${DEVICES[@]}"
            do
            ./assignment2 -n $size -m $size -b $block_size -d $device -p $ITERS -c -a -o
            done
        done
    done
done

#Larger block size timings

SIZES=(960 1920 3840 7680 15360)
BLOCKS=(32 64 128 256 512 992)
DEVICES=(0 1)
for size in "${SIZES[@]}"
do
    for block_size in "${BLOCKS[@]}"
    do
        for i in {1..3}
        do
            for device in "${DEVICES[@]}"
            do
            ./assignment2 -n $size -m $size -b $block_size -d $device -p $ITERS -c -a -o
            done
        done
    done
done

