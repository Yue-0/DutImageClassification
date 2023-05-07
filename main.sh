# shellcheck disable=SC2034
dir1=models/weights/teachers/
dir2=models/weights/students/

python data/convert.py data

teacher=ResNet50
echo ${teacher}
mkdir ${dir1}${teacher}
python train.py --save=${dir1}${teacher} --data=data --model=${teacher} --epochs=120 --batch_size=256 --weight_decay=1e-4 --lr=1e-1 --warmup=0
python inference.py --data data --models ${teacher} --weights ${dir1}${teacher}/best.pt --output results/teachers/${teacher}.csv

student=ResNet26
echo ${student}
mkdir ${dir2}${student}
python train.py --save=${dir2}${student} --data=data --model=${student} --epochs=180 --batch_size=256 --weight_decay=1e-4 --lr=1e-1 --warmup=0 --method=distillation --teacher=${teacher} --teacher_weights=${dir1}${teacher}/best.pt
python inference.py --data data --models ${student} --weights ${dir2}${student}/best.pt --output results/students/${student}.csv

student=RepVgg
echo ${student}
mkdir ${dir2}${student}
python train.py --save=${dir2}${student} --data=data --model=${student} --epochs=180 --batch_size=192 --weight_decay=5e-5 --lr=7.5e-2 --warmup=1 --method=distillation --teacher=${teacher} --teacher_weights=${dir1}${teacher}/best.pt
python inference.py --data data --models ${student} --weights ${dir2}${student}/best.pt --output results/students/${student}.csv

teacher=ConvNextTiny
echo ${teacher}
mkdir ${dir1}${teacher}
python train.py --save=${dir1}${teacher} --data=data --model=${teacher} --epochs=120 --batch_size=256 --weight_decay=1e-8 --lr=2e-3 --warmup=0
python inference.py --data data --models ${teacher} --weights ${dir1}${teacher}/best.pt --output results/teachers/${teacher}.csv

student=ConvNextNano
echo ${student}
mkdir ${dir2}${student}
python train.py --save=${dir2}${student} --data=data --model=${student} --epochs=180 --batch_size=192 --weight_decay=2e-2 --lr=2e-3 --warmup=10 --method=distillation --teacher=${teacher} --teacher_weights=${dir1}${teacher}/best.pt
python inference.py --data data --models ${student} --weights ${dir2}${student}/best.pt --output results/students/${student}.csv

teacher=SwinTransformerT
echo ${teacher}
mkdir ${dir1}${teacher}
python train.py --save=${dir1}${teacher} --data=data --model=${teacher} --epochs=120 --batch_size=224 --weight_decay=1e-8 --lr=2.5e-4 --warmup=0
python inference.py --data data --models ${teacher} --weights ${dir1}${teacher}/best.pt --output results/teachers/${teacher}.csv

student=SwinTransformerN
echo ${student}
mkdir ${dir2}${student}
python train.py --save=${dir2}${student} --data=data --model=${student} --epochs=180 --batch_size=128 --weight_decay=1e-3 --lr=4e-4 --warmup=10 --method=distillation --teacher=${teacher} --teacher_weights=${dir1}${teacher}/best.pt
python inference.py --data data --models ${student} --weights ${dir2}${student}/best.pt --output results/students/${student}.csv

python inference.py --data data --models ConvNextTiny ResNet50 SwinTransformerT --weights ${dir1}ConvNextTiny/best.pt ${dir1}ResNet50/best.pt ${dir1}SwinTransformerT/best.pt --output results/teachers/emsenble.csv
python inference.py --data data --models ConvNextNano ResNet26 SwinTransformerN RepVgg --weights ${dir2}ConvNextNano/best.pt ${dir2}ResNet26/best.pt ${dir2}SwinTransformerN/best.pt ${dir2}RepVgg/best.pt --output results/students/ensemble.csv
