?	???镲?????镲??!???镲??	o??Hd'@o??Hd'@!o??Hd'@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???镲??h"lxz???A?-????Yݵ?|г??*effff?W@)      =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateP?s???!??1DR?B@)Ǻ????1??Z?@@:Preprocessing2F
Iterator::ModelV-???!?d?ǧ>@)/?$???1??Q?26@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?I+???!??}A7@)??y?):??1?%J???2@:Preprocessing2U
Iterator::Model::ParallelMapV2????Mb??!??%J?? @)????Mb??1??%J?? @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip	?^)˰?!>?fVQ@)?g??s?u?1?2ru?h@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?J?4q?!?ǧ?L?@)?J?4q?1?ǧ?L?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mbp?!??%J??@)????Mbp?1??%J??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap@?߾???!8X%?=>D@)?~j?t?h?1)?8??^	@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 14.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9o??Hd'@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	h"lxz???h"lxz???!h"lxz???      ??!       "      ??!       *      ??!       2	?-?????-????!?-????:      ??!       B      ??!       J	ݵ?|г??ݵ?|г??!ݵ?|г??R      ??!       Z	ݵ?|г??ݵ?|г??!ݵ?|г??JCPU_ONLYYo??Hd'@b Y      Y@qP?"??wR@"?
both?Your program is POTENTIALLY input-bound because 14.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?73.8739% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 