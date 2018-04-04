# Sarkhans-Constitution-Generator

This repo is for educational and entertainment purposes only. 
// in fact it just for fun

we provide Sarkhan's constitution in constitution.txt
but you can have fun with your own country constitution by preprocess them into the same format (just have a look)
>>USE AT YOUR OWN RISK

>>Hint: it seem to be a good idea to replace some word to avoid legal issue 

to train simply run


python train.py [textfile]


E.g.    python train.py constitution.txt

to draft some Constitution just


python draft.py [Constitution length in word] --seed [seed word]
(seed is optional)

E.g.    python draft.py 3000 --seed ประชาชน



# Extra
Tokenizer in thai-word-segmentation[Jousimo et al] submodule might be helpful if you want to process your own constitution.
for more information see [This blog post](https://sertiscorp.com/thai-word-segmentation-with-bi-directional_rnn/)

the generator are modified from [This repo](https://github.com/udacity/deep-learning/tree/master/tv-script-generation)
