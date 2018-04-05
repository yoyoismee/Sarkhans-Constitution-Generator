# Sarkhans-Constitution-Generator

This repo is for educational and entertainment purposes only. 

// in fact it just for fun

## Requirements
* python 
* tensorflow 
* pickle

## Usage
we provide Sarkhan's constitution in constitution.txt
but you can have fun with your own country constitution by preprocess them into the same format just [have a look at](https://github.com/yoyoismee/Sarkhans-Constitution-Generator/blob/master/constitution.txt)
>>USE AT YOUR OWN RISK

>>Hint: it seem to be a good idea to replace some word to avoid legal issue 

to train simply run


python train.py [textfile]


E.g.    python train.py constitution.txt

to draft some Constitution just


python draft.py [Constitution length in word] --seed [seed word]
(seed is optional)

E.g.    python draft.py 3000 --seed ประชาชน

## sample resault

"ในกรณีที่ผู้ตรวจการแผ่นดินของรัฐสภาเห็นว่าบทบัญญัติแห่งกฎหมาย และยังไม่ส่งความเห็นต้องพิจารณาให้เสร็จภายในสามสิบวัน ทั้งนี้ เว้นแต่วุฒิสภาจะได้ลงมติให้ขยายเวลาออกไปเป็นกรณีพิเศษซึ่งต้องไม่เกินสองร้อยสี่สิบวันนับแต่วันประกาศใช้รัฐธรรมนูญนี้ ให้ดำเนินการตรากฎหมายประกอบรัฐธรรมนูญดังต่อไปนี้ให้แล้วเสร็จ(๑) มิให้นำบทบัญญัติมาตรา ๒๕๕ (๒) มาใช้บังคับโดยอนุโลม ในกรณีนี้ ให้ส่งความเห็นเช่นว่ากระทำการอันจำเป็นเฉพาะเพื่อการผูกขาดความมั่นคงของรัฐ หรือรัฐวิสาหกิจ ในการปฏิบัติหน้าที่ ให้ดำเนินการต่อไป”


## Extra
Tokenizer in thai-word-segmentation[Jousimo et al] submodule might be helpful if you want to process your own constitution.
for more information see [this blog post](https://sertiscorp.com/thai-word-segmentation-with-bi-directional_rnn/) or [this repo](https://github.com/sertiscorp/thai-word-segmentation)

the generator are modified from [This repo](https://github.com/udacity/deep-learning/tree/master/tv-script-generation)
