rm inference.txt
rm test_single_image.txt

for ((i=1; i<=20; i++))
do
    experiments/scripts/inference.sh >> inference.txt
    experiments/scripts/test_single_image.sh >> test_single_image.txt
done
