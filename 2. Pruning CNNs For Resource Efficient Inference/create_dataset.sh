mkdir test
mkdir -p test/dogs
mkdir -p test/cats
mkdir train
mkdir -p train/dogs
mkdir -p train/cats

#download dogs_vs_cats from kaggle and unzip
kaggle competitions download -c dogs-vs-cats
unzip dogs-vs-cats.zip -d dogs-vs-cats

#1000 images in train and 400 images in test

for i in $(seq 1 1000)
do
cp dogs-vs-cats/train/train/cat.$i.jpg train/cats/
cp dogs-vs-cats/train/train/dog.$i.jpg train/dogs/
done



for i in $(seq 1001 1400)
do
cp dogs-vs-cats/train/train/cat.$i.jpg test/cats/
cp dogs-vs-cats/train/train/dog.$i.jpg test/dogs/
done
