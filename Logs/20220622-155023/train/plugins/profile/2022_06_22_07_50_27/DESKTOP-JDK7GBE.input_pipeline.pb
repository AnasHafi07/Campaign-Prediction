	鷯???鷯???!鷯???	???_@???_@!???_@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$鷯????(??0??A?ڊ?e???Y?(??0??*	     `S@2F
Iterator::Model2??%䃞?!:??s?9C@)A??ǘ???1??RJ)?<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?HP???!{????{?@)n????1K)??RJ9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate? ?	???!?{????3@)tF??_??1??Zk??.@:Preprocessing2U
Iterator::Model::ParallelMapV2ŏ1w-!?!?s?9??#@)ŏ1w-!?1?s?9??#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??	h"l??!?c?1?N@)??_?Lu?1?Zk???@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?s?!?c?1?@)a2U0*?s?1?c?1?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey?&1?l?!B!?@)y?&1?l?1B!?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?:pΈ??![k???Z7@)??_vOf?1?{????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???_@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(??0???(??0??!?(??0??      ??!       "      ??!       *      ??!       2	?ڊ?e????ڊ?e???!?ڊ?e???:      ??!       B      ??!       J	?(??0???(??0??!?(??0??R      ??!       Z	?(??0???(??0??!?(??0??JCPU_ONLYY???_@b 