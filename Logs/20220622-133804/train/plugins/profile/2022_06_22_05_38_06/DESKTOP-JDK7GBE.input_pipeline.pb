	??n??????n????!??n????	?C8?C8@?C8?C8@!?C8?C8@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??n??????H?}??A?ܵ?|???YM?J???*	33333sO@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???&??!?L?Ϻ=@)???_vO??1??????7@:Preprocessing2F
Iterator::Model??_vO??!@??i?+A@)V-???1"|?Kn7@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip0L?
F%??!?2?9jP@)?j+??݃?1?????.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?I+???!?L???|1@)vq?-??1??N?)@:Preprocessing2U
Iterator::Model::ParallelMapV2?ZӼ?}?!?p?U?&@)?ZӼ?}?1?p?U?&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?q????o?!?P' ?@)?q????o?1?P' ?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea??+ei?!],稽?@)a??+ei?1],稽?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?{??Pk??!??=?4@)ŏ1w-!_?1?L?S*@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t14.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?C8?C8@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??H?}????H?}??!??H?}??      ??!       "      ??!       *      ??!       2	?ܵ?|????ܵ?|???!?ܵ?|???:      ??!       B      ??!       J	M?J???M?J???!M?J???R      ??!       Z	M?J???M?J???!M?J???JCPU_ONLYY?C8?C8@b 