cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
#root_dir=$cur_dir/../..
#root_dir="../.."
root_dir="/home/ssd/caffe"

cd $root_dir

redo=1
#data_root_dir="$HOME/data/VOCdevkit"
#dataset_name="VOC0712"
#data_root_dir="../VOCdevkit"
data_root_dir="$root_dir/data/VOCdevkit"
dataset_name="banner"
#mapfile="$root_dir/data/$dataset_name/labelmap_voc.prototxt"
mapfile="$root_dir/data/$dataset_name/labelmap_voc_banner.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test trainval
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $root_dir/data/$dataset_name/$subset.txt $data_root_dir/$dataset_name/$db/$dataset_name"_"$subset"_"$db examples/$dataset_name
done
