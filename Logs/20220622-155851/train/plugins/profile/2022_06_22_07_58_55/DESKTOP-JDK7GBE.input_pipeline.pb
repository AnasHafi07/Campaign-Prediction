	D????9??D????9??!D????9??	=l??? @=l??? @!=l??? @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$D????9??Z??ڊ???A%u???YbX9?ȶ?*	?????ye@2F
Iterator::ModelHP?sײ?!?????kE@)	?^)˰?1KL -?C@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate*??Dذ?!n??q&C@)?q??????1"
?{ )B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatHP?s??!?%S??*@)Έ?????1<?ѓ0?%@:Preprocessing2U
Iterator::Model::ParallelMapV2????Mb??!7+}?U?@)????Mb??17+}?U?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipk?w??#??!Npj?L@)?<,Ԛ?}?1o7??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor"??u??q?!t???@)"??u??q?1t???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_?Q?k?!v?Ԃ+???)_?Q?k?1v?Ԃ+???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?? ?rh??!???1[?C@)/n??b?1???6+}??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.3% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t21.8 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9>l??? @>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Z??ڊ???Z??ڊ???!Z??ڊ???      ??!       "      ??!       *      ??!       2	%u???%u???!%u???:      ??!       B      ??!       J	bX9?ȶ?bX9?ȶ?!bX9?ȶ?R      ??!       Z	bX9?ȶ?bX9?ȶ?!bX9?ȶ?JCPU_ONLYY>l??? @b 