ΆΔ
Ρ£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
Ύ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ρ
|
hidden_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
η* 
shared_namehidden_1/kernel
u
#hidden_1/kernel/Read/ReadVariableOpReadVariableOphidden_1/kernel* 
_output_shapes
:
η*
dtype0
s
hidden_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden_1/bias
l
!hidden_1/bias/Read/ReadVariableOpReadVariableOphidden_1/bias*
_output_shapes	
:*
dtype0
|
hidden_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namehidden_2/kernel
u
#hidden_2/kernel/Read/ReadVariableOpReadVariableOphidden_2/kernel* 
_output_shapes
:
*
dtype0
s
hidden_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namehidden_2/bias
l
!hidden_2/bias/Read/ReadVariableOpReadVariableOphidden_2/bias*
_output_shapes	
:*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/hidden_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
η*'
shared_nameAdam/hidden_1/kernel/m

*Adam/hidden_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/m* 
_output_shapes
:
η*
dtype0

Adam/hidden_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/hidden_1/bias/m
z
(Adam/hidden_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_1/bias/m*
_output_shapes	
:*
dtype0

Adam/hidden_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/hidden_2/kernel/m

*Adam/hidden_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/hidden_2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/hidden_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/hidden_2/bias/m
z
(Adam/hidden_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/hidden_2/bias/m*
_output_shapes	
:*
dtype0

Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/output/kernel/m
~
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes
:	*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0

Adam/hidden_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
η*'
shared_nameAdam/hidden_1/kernel/v

*Adam/hidden_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/kernel/v* 
_output_shapes
:
η*
dtype0

Adam/hidden_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/hidden_1/bias/v
z
(Adam/hidden_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_1/bias/v*
_output_shapes	
:*
dtype0

Adam/hidden_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/hidden_2/kernel/v

*Adam/hidden_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/hidden_2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/hidden_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/hidden_2/bias/v
z
(Adam/hidden_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/hidden_2/bias/v*
_output_shapes	
:*
dtype0

Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/output/kernel/v
~
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes
:	*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Κ'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*'
valueϋ&Bψ& Bρ&

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
h

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
¬
&iter

'beta_1

(beta_2
	)decay
*learning_ratemNmOmPmQ mR!mSvTvUvVvW vX!vY
*
0
1
2
3
 4
!5
 
*
0
1
2
3
 4
!5
­
+layer_regularization_losses
,non_trainable_variables
	variables

-layers
regularization_losses
.layer_metrics
/metrics
	trainable_variables
 
[Y
VARIABLE_VALUEhidden_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
0layer_regularization_losses
1non_trainable_variables
	variables

2layers
trainable_variables
3layer_metrics
4metrics
regularization_losses
 
 
 
­
5layer_regularization_losses
6non_trainable_variables
	variables

7layers
trainable_variables
8layer_metrics
9metrics
regularization_losses
[Y
VARIABLE_VALUEhidden_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEhidden_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
:layer_regularization_losses
;non_trainable_variables
	variables

<layers
trainable_variables
=layer_metrics
>metrics
regularization_losses
 
 
 
­
?layer_regularization_losses
@non_trainable_variables
	variables

Alayers
trainable_variables
Blayer_metrics
Cmetrics
regularization_losses
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

 0
!1

 0
!1
 
­
Dlayer_regularization_losses
Enon_trainable_variables
"	variables

Flayers
#trainable_variables
Glayer_metrics
Hmetrics
$regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
#
0
1
2
3
4
 

I0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Jtotal
	Kcount
L	variables
M	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

L	variables
~|
VARIABLE_VALUEAdam/hidden_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/hidden_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/hidden_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_hidden_1_inputPlaceholder*(
_output_shapes
:?????????η*
dtype0*
shape:?????????η
‘
StatefulPartitionedCallStatefulPartitionedCallserving_default_hidden_1_inputhidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biasoutput/kerneloutput/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_163987
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
γ	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#hidden_1/kernel/Read/ReadVariableOp!hidden_1/bias/Read/ReadVariableOp#hidden_2/kernel/Read/ReadVariableOp!hidden_2/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp*Adam/hidden_1/kernel/m/Read/ReadVariableOp(Adam/hidden_1/bias/m/Read/ReadVariableOp*Adam/hidden_2/kernel/m/Read/ReadVariableOp(Adam/hidden_2/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp*Adam/hidden_1/kernel/v/Read/ReadVariableOp(Adam/hidden_1/bias/v/Read/ReadVariableOp*Adam/hidden_2/kernel/v/Read/ReadVariableOp(Adam/hidden_2/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_164298
κ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamehidden_1/kernelhidden_1/biashidden_2/kernelhidden_2/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/hidden_1/kernel/mAdam/hidden_1/bias/mAdam/hidden_2/kernel/mAdam/hidden_2/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/hidden_1/kernel/vAdam/hidden_1/bias/vAdam/hidden_2/kernel/vAdam/hidden_2/bias/vAdam/output/kernel/vAdam/output/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_164383μ¨
Κ
a
C__inference_dropout_layer_call_and_return_conditional_losses_163765

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
?*
ς
F__inference_sequential_layer_call_and_return_conditional_losses_164027

inputs+
'hidden_1_matmul_readvariableop_resource,
(hidden_1_biasadd_readvariableop_resource+
'hidden_2_matmul_readvariableop_resource,
(hidden_2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityͺ
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
η*
dtype02 
hidden_1/MatMul/ReadVariableOp
hidden_1/MatMulMatMulinputs&hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden_1/MatMul¨
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
hidden_1/BiasAdd/ReadVariableOp¦
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden_1/BiasAddt
hidden_1/ReluReluhidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden_1/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/dropout/Const‘
dropout/dropout/MulMulhidden_1/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/dropout/Muly
dropout/dropout/ShapeShapehidden_1/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeΝ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dropout/dropout/GreaterEqual/yί
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/dropout/Mul_1ͺ
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
hidden_2/MatMul/ReadVariableOp’
hidden_2/MatMulMatMuldropout/dropout/Mul_1:z:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden_2/MatMul¨
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
hidden_2/BiasAdd/ReadVariableOp¦
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden_2/BiasAddt
hidden_2/ReluReluhidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden_2/Reluw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout_1/dropout/Const§
dropout_1/dropout/MulMulhidden_2/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout_1/dropout/Mul}
dropout_1/dropout/ShapeShapehidden_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeΣ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 dropout_1/dropout/GreaterEqual/yη
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2 
dropout_1/dropout/GreaterEqual
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout_1/dropout/Cast£
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout_1/dropout/Mul_1£
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldropout_1/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul‘
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddk
IdentityIdentityoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η:::::::P L
(
_output_shapes
:?????????η
 
_user_specified_nameinputs


F__inference_sequential_layer_call_and_return_conditional_losses_163907

inputs
hidden_1_163889
hidden_1_163891
hidden_2_163895
hidden_2_163897
output_163901
output_163903
identity’dropout/StatefulPartitionedCall’!dropout_1/StatefulPartitionedCall’ hidden_1/StatefulPartitionedCall’ hidden_2/StatefulPartitionedCall’output/StatefulPartitionedCall
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_1_163889hidden_1_163891*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_1637322"
 hidden_1/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1637602!
dropout/StatefulPartitionedCall·
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0hidden_2_163895hidden_2_163897*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_2_layer_call_and_return_conditional_losses_1637892"
 hidden_2/StatefulPartitionedCall΅
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1638172#
!dropout_1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0output_163901output_163903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1638452 
output/StatefulPartitionedCall¨
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
(
_output_shapes
:?????????η
 
_user_specified_nameinputs

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_164166

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape΅
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yΏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ν
ς
F__inference_sequential_layer_call_and_return_conditional_losses_164053

inputs+
'hidden_1_matmul_readvariableop_resource,
(hidden_1_biasadd_readvariableop_resource+
'hidden_2_matmul_readvariableop_resource,
(hidden_2_biasadd_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identityͺ
hidden_1/MatMul/ReadVariableOpReadVariableOp'hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
η*
dtype02 
hidden_1/MatMul/ReadVariableOp
hidden_1/MatMulMatMulinputs&hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden_1/MatMul¨
hidden_1/BiasAdd/ReadVariableOpReadVariableOp(hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
hidden_1/BiasAdd/ReadVariableOp¦
hidden_1/BiasAddBiasAddhidden_1/MatMul:product:0'hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden_1/BiasAddt
hidden_1/ReluReluhidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden_1/Relu
dropout/IdentityIdentityhidden_1/Relu:activations:0*
T0*(
_output_shapes
:?????????2
dropout/Identityͺ
hidden_2/MatMul/ReadVariableOpReadVariableOp'hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02 
hidden_2/MatMul/ReadVariableOp’
hidden_2/MatMulMatMuldropout/Identity:output:0&hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden_2/MatMul¨
hidden_2/BiasAdd/ReadVariableOpReadVariableOp(hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
hidden_2/BiasAdd/ReadVariableOp¦
hidden_2/BiasAddBiasAddhidden_2/MatMul:product:0'hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
hidden_2/BiasAddt
hidden_2/ReluReluhidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
hidden_2/Relu
dropout_1/IdentityIdentityhidden_2/Relu:activations:0*
T0*(
_output_shapes
:?????????2
dropout_1/Identity£
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldropout_1/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/MatMul‘
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
output/BiasAddk
IdentityIdentityoutput/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η:::::::P L
(
_output_shapes
:?????????η
 
_user_specified_nameinputs
²
¬
D__inference_hidden_1_layer_call_and_return_conditional_losses_164098

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
η*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????η:::P L
(
_output_shapes
:?????????η
 
_user_specified_nameinputs
Φ

!__inference__wrapped_model_163717
hidden_1_input6
2sequential_hidden_1_matmul_readvariableop_resource7
3sequential_hidden_1_biasadd_readvariableop_resource6
2sequential_hidden_2_matmul_readvariableop_resource7
3sequential_hidden_2_biasadd_readvariableop_resource4
0sequential_output_matmul_readvariableop_resource5
1sequential_output_biasadd_readvariableop_resource
identityΛ
)sequential/hidden_1/MatMul/ReadVariableOpReadVariableOp2sequential_hidden_1_matmul_readvariableop_resource* 
_output_shapes
:
η*
dtype02+
)sequential/hidden_1/MatMul/ReadVariableOpΈ
sequential/hidden_1/MatMulMatMulhidden_1_input1sequential/hidden_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
sequential/hidden_1/MatMulΙ
*sequential/hidden_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_hidden_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*sequential/hidden_1/BiasAdd/ReadVariableOp?
sequential/hidden_1/BiasAddBiasAdd$sequential/hidden_1/MatMul:product:02sequential/hidden_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
sequential/hidden_1/BiasAdd
sequential/hidden_1/ReluRelu$sequential/hidden_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
sequential/hidden_1/Relu‘
sequential/dropout/IdentityIdentity&sequential/hidden_1/Relu:activations:0*
T0*(
_output_shapes
:?????????2
sequential/dropout/IdentityΛ
)sequential/hidden_2/MatMul/ReadVariableOpReadVariableOp2sequential_hidden_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02+
)sequential/hidden_2/MatMul/ReadVariableOpΞ
sequential/hidden_2/MatMulMatMul$sequential/dropout/Identity:output:01sequential/hidden_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
sequential/hidden_2/MatMulΙ
*sequential/hidden_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_hidden_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02,
*sequential/hidden_2/BiasAdd/ReadVariableOp?
sequential/hidden_2/BiasAddBiasAdd$sequential/hidden_2/MatMul:product:02sequential/hidden_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
sequential/hidden_2/BiasAdd
sequential/hidden_2/ReluRelu$sequential/hidden_2/BiasAdd:output:0*
T0*(
_output_shapes
:?????????2
sequential/hidden_2/Relu₯
sequential/dropout_1/IdentityIdentity&sequential/hidden_2/Relu:activations:0*
T0*(
_output_shapes
:?????????2
sequential/dropout_1/IdentityΔ
'sequential/output/MatMul/ReadVariableOpReadVariableOp0sequential_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02)
'sequential/output/MatMul/ReadVariableOpΙ
sequential/output/MatMulMatMul&sequential/dropout_1/Identity:output:0/sequential/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/MatMulΒ
(sequential/output/BiasAdd/ReadVariableOpReadVariableOp1sequential_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/output/BiasAdd/ReadVariableOpΙ
sequential/output/BiasAddBiasAdd"sequential/output/MatMul:product:00sequential/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/output/BiasAddv
IdentityIdentity"sequential/output/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η:::::::X T
(
_output_shapes
:?????????η
(
_user_specified_namehidden_1_input

d
E__inference_dropout_1_layer_call_and_return_conditional_losses_163817

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape΅
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yΏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

b
C__inference_dropout_layer_call_and_return_conditional_losses_163760

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape΅
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yΏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs

F
*__inference_dropout_1_layer_call_fn_164181

inputs
identityΔ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1638222
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
²
¬
D__inference_hidden_2_layer_call_and_return_conditional_losses_164145

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
²
¬
D__inference_hidden_2_layer_call_and_return_conditional_losses_163789

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ϋ
Δ
+__inference_sequential_layer_call_fn_163960
hidden_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallhidden_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1639452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:?????????η
(
_user_specified_namehidden_1_input
γ
Ό
+__inference_sequential_layer_call_fn_164070

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCallͺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1639072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????η
 
_user_specified_nameinputs
Ξ
ͺ
B__inference_output_layer_call_and_return_conditional_losses_164191

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
²
¬
D__inference_hidden_1_layer_call_and_return_conditional_losses_163732

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
η*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????η:::P L
(
_output_shapes
:?????????η
 
_user_specified_nameinputs
Κ
a
C__inference_dropout_layer_call_and_return_conditional_losses_164124

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Μ
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_163822

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
©

F__inference_sequential_layer_call_and_return_conditional_losses_163862
hidden_1_input
hidden_1_163743
hidden_1_163745
hidden_2_163800
hidden_2_163802
output_163856
output_163858
identity’dropout/StatefulPartitionedCall’!dropout_1/StatefulPartitionedCall’ hidden_1/StatefulPartitionedCall’ hidden_2/StatefulPartitionedCall’output/StatefulPartitionedCall
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallhidden_1_inputhidden_1_163743hidden_1_163745*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_1637322"
 hidden_1/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall)hidden_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1637602!
dropout/StatefulPartitionedCall·
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0hidden_2_163800hidden_2_163802*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_2_layer_call_and_return_conditional_losses_1637892"
 hidden_2/StatefulPartitionedCall΅
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)hidden_2/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1638172#
!dropout_1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0output_163856output_163858*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1638452 
output/StatefulPartitionedCall¨
IdentityIdentity'output/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η::::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
(
_output_shapes
:?????????η
(
_user_specified_namehidden_1_input
γ
Ό
+__inference_sequential_layer_call_fn_164087

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCallͺ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1639452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????η
 
_user_specified_nameinputs

b
C__inference_dropout_layer_call_and_return_conditional_losses_164119

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:?????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape΅
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:?????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
dropout/GreaterEqual/yΏ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:?????????2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:?????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:?????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
€
c
*__inference_dropout_1_layer_call_fn_164176

inputs
identity’StatefulPartitionedCallά
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1638172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
;
?

__inference__traced_save_164298
file_prefix.
*savev2_hidden_1_kernel_read_readvariableop,
(savev2_hidden_1_bias_read_readvariableop.
*savev2_hidden_2_kernel_read_readvariableop,
(savev2_hidden_2_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop5
1savev2_adam_hidden_1_kernel_m_read_readvariableop3
/savev2_adam_hidden_1_bias_m_read_readvariableop5
1savev2_adam_hidden_2_kernel_m_read_readvariableop3
/savev2_adam_hidden_2_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop5
1savev2_adam_hidden_1_kernel_v_read_readvariableop3
/savev2_adam_hidden_1_bias_v_read_readvariableop5
1savev2_adam_hidden_2_kernel_v_read_readvariableop3
/savev2_adam_hidden_2_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_76d95e73b1184801b07823571d0f76eb/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B₯B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesΌ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices±

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_hidden_1_kernel_read_readvariableop(savev2_hidden_1_bias_read_readvariableop*savev2_hidden_2_kernel_read_readvariableop(savev2_hidden_2_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop1savev2_adam_hidden_1_kernel_m_read_readvariableop/savev2_adam_hidden_1_bias_m_read_readvariableop1savev2_adam_hidden_2_kernel_m_read_readvariableop/savev2_adam_hidden_2_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop1savev2_adam_hidden_1_kernel_v_read_readvariableop/savev2_adam_hidden_1_bias_v_read_readvariableop1savev2_adam_hidden_2_kernel_v_read_readvariableop/savev2_adam_hidden_2_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	2
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Μ
_input_shapesΊ
·: :
η::
::	:: : : : : : : :
η::
::	::
η::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
η:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
η:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::&"
 
_output_shapes
:
η:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: 

D
(__inference_dropout_layer_call_fn_164134

inputs
identityΒ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1637652
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
k
κ
"__inference__traced_restore_164383
file_prefix$
 assignvariableop_hidden_1_kernel$
 assignvariableop_1_hidden_1_bias&
"assignvariableop_2_hidden_2_kernel$
 assignvariableop_3_hidden_2_bias$
 assignvariableop_4_output_kernel"
assignvariableop_5_output_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count.
*assignvariableop_13_adam_hidden_1_kernel_m,
(assignvariableop_14_adam_hidden_1_bias_m.
*assignvariableop_15_adam_hidden_2_kernel_m,
(assignvariableop_16_adam_hidden_2_bias_m,
(assignvariableop_17_adam_output_kernel_m*
&assignvariableop_18_adam_output_bias_m.
*assignvariableop_19_adam_hidden_1_kernel_v,
(assignvariableop_20_adam_hidden_1_bias_v.
*assignvariableop_21_adam_hidden_2_kernel_v,
(assignvariableop_22_adam_hidden_2_bias_v,
(assignvariableop_23_adam_output_kernel_v*
&assignvariableop_24_adam_output_bias_v
identity_26’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9¦
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B₯B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesΒ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices­
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_hidden_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1₯
AssignVariableOp_1AssignVariableOp assignvariableop_1_hidden_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2§
AssignVariableOp_2AssignVariableOp"assignvariableop_2_hidden_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3₯
AssignVariableOp_3AssignVariableOp assignvariableop_3_hidden_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4₯
AssignVariableOp_4AssignVariableOp assignvariableop_4_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6‘
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9’
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11‘
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12‘
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13²
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_hidden_1_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14°
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_hidden_1_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15²
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_hidden_2_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16°
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_hidden_2_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17°
AssignVariableOp_17AssignVariableOp(assignvariableop_17_adam_output_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_adam_output_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19²
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_hidden_1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20°
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_hidden_1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21²
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_hidden_2_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22°
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_hidden_2_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23°
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_output_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp&assignvariableop_24_adam_output_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_249
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_25χ
Identity_26IdentityIdentity_25:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_26"#
identity_26Identity_26:output:0*y
_input_shapesh
f: :::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ΰ
~
)__inference_hidden_2_layer_call_fn_164154

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallυ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_2_layer_call_and_return_conditional_losses_1637892
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ο
½
$__inference_signature_wrapper_163987
hidden_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallhidden_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_1637172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:?????????η
(
_user_specified_namehidden_1_input
Μ
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_164171

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:?????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:?????????2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:?????????:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
―
Ξ
F__inference_sequential_layer_call_and_return_conditional_losses_163883
hidden_1_input
hidden_1_163865
hidden_1_163867
hidden_2_163871
hidden_2_163873
output_163877
output_163879
identity’ hidden_1/StatefulPartitionedCall’ hidden_2/StatefulPartitionedCall’output/StatefulPartitionedCall
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallhidden_1_inputhidden_1_163865hidden_1_163867*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_1637322"
 hidden_1/StatefulPartitionedCallυ
dropout/PartitionedCallPartitionedCall)hidden_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1637652
dropout/PartitionedCall―
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0hidden_2_163871hidden_2_163873*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_2_layer_call_and_return_conditional_losses_1637892"
 hidden_2/StatefulPartitionedCallϋ
dropout_1/PartitionedCallPartitionedCall)hidden_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1638222
dropout_1/PartitionedCall¦
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0output_163877output_163879*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1638452 
output/StatefulPartitionedCallβ
IdentityIdentity'output/StatefulPartitionedCall:output:0!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η::::::2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
(
_output_shapes
:?????????η
(
_user_specified_namehidden_1_input

Ζ
F__inference_sequential_layer_call_and_return_conditional_losses_163945

inputs
hidden_1_163927
hidden_1_163929
hidden_2_163933
hidden_2_163935
output_163939
output_163941
identity’ hidden_1/StatefulPartitionedCall’ hidden_2/StatefulPartitionedCall’output/StatefulPartitionedCall
 hidden_1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_1_163927hidden_1_163929*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_1637322"
 hidden_1/StatefulPartitionedCallυ
dropout/PartitionedCallPartitionedCall)hidden_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1637652
dropout/PartitionedCall―
 hidden_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0hidden_2_163933hidden_2_163935*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_2_layer_call_and_return_conditional_losses_1637892"
 hidden_2/StatefulPartitionedCallϋ
dropout_1/PartitionedCallPartitionedCall)hidden_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_1638222
dropout_1/PartitionedCall¦
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0output_163939output_163941*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1638452 
output/StatefulPartitionedCallβ
IdentityIdentity'output/StatefulPartitionedCall:output:0!^hidden_1/StatefulPartitionedCall!^hidden_2/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η::::::2D
 hidden_1/StatefulPartitionedCall hidden_1/StatefulPartitionedCall2D
 hidden_2/StatefulPartitionedCall hidden_2/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:P L
(
_output_shapes
:?????????η
 
_user_specified_nameinputs
 
a
(__inference_dropout_layer_call_fn_164129

inputs
identity’StatefulPartitionedCallΪ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_1637602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:?????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ξ
ͺ
B__inference_output_layer_call_and_return_conditional_losses_163845

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????:::P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϊ
|
'__inference_output_layer_call_fn_164200

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallς
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_1638452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ϋ
Δ
+__inference_sequential_layer_call_fn_163922
hidden_1_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity’StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallhidden_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_1639072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:?????????η::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
(
_output_shapes
:?????????η
(
_user_specified_namehidden_1_input
ΰ
~
)__inference_hidden_1_layer_call_fn_164107

inputs
unknown
	unknown_0
identity’StatefulPartitionedCallυ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_hidden_1_layer_call_and_return_conditional_losses_1637322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:?????????η::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????η
 
_user_specified_nameinputs"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Έ
serving_default€
J
hidden_1_input8
 serving_default_hidden_1_input:0?????????η:
output0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?¨
'
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	optimizer
	variables
regularization_losses
	trainable_variables

	keras_api

signatures
Z_default_save_signature
*[&call_and_return_all_conditional_losses
\__call__"΅$
_tf_keras_sequential${"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 999]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hidden_1_input"}}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 999]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 999}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 999]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 999]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hidden_1_input"}}, {"class_name": "Dense", "config": {"name": "hidden_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 999]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_absolute_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999974752427e-07, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
κ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*]&call_and_return_all_conditional_losses
^__call__"Ε
_tf_keras_layer«{"class_name": "Dense", "name": "hidden_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 999]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 999]}, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 999}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 999]}}
α
	variables
trainable_variables
regularization_losses
	keras_api
*_&call_and_return_all_conditional_losses
`__call__"?
_tf_keras_layerΈ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
υ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
*a&call_and_return_all_conditional_losses
b__call__"Π
_tf_keras_layerΆ{"class_name": "Dense", "name": "hidden_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hidden_2", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ε
	variables
trainable_variables
regularization_losses
	keras_api
*c&call_and_return_all_conditional_losses
d__call__"Φ
_tf_keras_layerΌ{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
ρ

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
*e&call_and_return_all_conditional_losses
f__call__"Μ
_tf_keras_layer²{"class_name": "Dense", "name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
Ώ
&iter

'beta_1

(beta_2
	)decay
*learning_ratemNmOmPmQ mR!mSvTvUvVvW vX!vY"
	optimizer
J
0
1
2
3
 4
!5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
Κ
+layer_regularization_losses
,non_trainable_variables
	variables

-layers
regularization_losses
.layer_metrics
/metrics
	trainable_variables
\__call__
Z_default_save_signature
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
,
gserving_default"
signature_map
#:!
η2hidden_1/kernel
:2hidden_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
0layer_regularization_losses
1non_trainable_variables
	variables

2layers
trainable_variables
3layer_metrics
4metrics
regularization_losses
^__call__
*]&call_and_return_all_conditional_losses
&]"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
5layer_regularization_losses
6non_trainable_variables
	variables

7layers
trainable_variables
8layer_metrics
9metrics
regularization_losses
`__call__
*_&call_and_return_all_conditional_losses
&_"call_and_return_conditional_losses"
_generic_user_object
#:!
2hidden_2/kernel
:2hidden_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
:layer_regularization_losses
;non_trainable_variables
	variables

<layers
trainable_variables
=layer_metrics
>metrics
regularization_losses
b__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
?layer_regularization_losses
@non_trainable_variables
	variables

Alayers
trainable_variables
Blayer_metrics
Cmetrics
regularization_losses
d__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
 :	2output/kernel
:2output/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dlayer_regularization_losses
Enon_trainable_variables
"	variables

Flayers
#trainable_variables
Glayer_metrics
Hmetrics
$regularization_losses
f__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_dict_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
»
	Jtotal
	Kcount
L	variables
M	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
.
J0
K1"
trackable_list_wrapper
-
L	variables"
_generic_user_object
(:&
η2Adam/hidden_1/kernel/m
!:2Adam/hidden_1/bias/m
(:&
2Adam/hidden_2/kernel/m
!:2Adam/hidden_2/bias/m
%:#	2Adam/output/kernel/m
:2Adam/output/bias/m
(:&
η2Adam/hidden_1/kernel/v
!:2Adam/hidden_1/bias/v
(:&
2Adam/hidden_2/kernel/v
!:2Adam/hidden_2/bias/v
%:#	2Adam/output/kernel/v
:2Adam/output/bias/v
η2δ
!__inference__wrapped_model_163717Ύ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *.’+
)&
hidden_1_input?????????η
ζ2γ
F__inference_sequential_layer_call_and_return_conditional_losses_163883
F__inference_sequential_layer_call_and_return_conditional_losses_164027
F__inference_sequential_layer_call_and_return_conditional_losses_164053
F__inference_sequential_layer_call_and_return_conditional_losses_163862ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ϊ2χ
+__inference_sequential_layer_call_fn_163922
+__inference_sequential_layer_call_fn_164070
+__inference_sequential_layer_call_fn_164087
+__inference_sequential_layer_call_fn_163960ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
D__inference_hidden_1_layer_call_and_return_conditional_losses_164098’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_hidden_1_layer_call_fn_164107’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Δ2Α
C__inference_dropout_layer_call_and_return_conditional_losses_164119
C__inference_dropout_layer_call_and_return_conditional_losses_164124΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
(__inference_dropout_layer_call_fn_164129
(__inference_dropout_layer_call_fn_164134΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ξ2λ
D__inference_hidden_2_layer_call_and_return_conditional_losses_164145’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Σ2Π
)__inference_hidden_2_layer_call_fn_164154’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Θ2Ε
E__inference_dropout_1_layer_call_and_return_conditional_losses_164171
E__inference_dropout_1_layer_call_and_return_conditional_losses_164166΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
2
*__inference_dropout_1_layer_call_fn_164181
*__inference_dropout_1_layer_call_fn_164176΄
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
μ2ι
B__inference_output_layer_call_and_return_conditional_losses_164191’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
Ρ2Ξ
'__inference_output_layer_call_fn_164200’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:B8
$__inference_signature_wrapper_163987hidden_1_input
!__inference__wrapped_model_163717s !8’5
.’+
)&
hidden_1_input?????????η
ͺ "/ͺ,
*
output 
output?????????§
E__inference_dropout_1_layer_call_and_return_conditional_losses_164166^4’1
*’'
!
inputs?????????
p
ͺ "&’#

0?????????
 §
E__inference_dropout_1_layer_call_and_return_conditional_losses_164171^4’1
*’'
!
inputs?????????
p 
ͺ "&’#

0?????????
 
*__inference_dropout_1_layer_call_fn_164176Q4’1
*’'
!
inputs?????????
p
ͺ "?????????
*__inference_dropout_1_layer_call_fn_164181Q4’1
*’'
!
inputs?????????
p 
ͺ "?????????₯
C__inference_dropout_layer_call_and_return_conditional_losses_164119^4’1
*’'
!
inputs?????????
p
ͺ "&’#

0?????????
 ₯
C__inference_dropout_layer_call_and_return_conditional_losses_164124^4’1
*’'
!
inputs?????????
p 
ͺ "&’#

0?????????
 }
(__inference_dropout_layer_call_fn_164129Q4’1
*’'
!
inputs?????????
p
ͺ "?????????}
(__inference_dropout_layer_call_fn_164134Q4’1
*’'
!
inputs?????????
p 
ͺ "?????????¦
D__inference_hidden_1_layer_call_and_return_conditional_losses_164098^0’-
&’#
!
inputs?????????η
ͺ "&’#

0?????????
 ~
)__inference_hidden_1_layer_call_fn_164107Q0’-
&’#
!
inputs?????????η
ͺ "?????????¦
D__inference_hidden_2_layer_call_and_return_conditional_losses_164145^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 ~
)__inference_hidden_2_layer_call_fn_164154Q0’-
&’#
!
inputs?????????
ͺ "?????????£
B__inference_output_layer_call_and_return_conditional_losses_164191] !0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????
 {
'__inference_output_layer_call_fn_164200P !0’-
&’#
!
inputs?????????
ͺ "?????????»
F__inference_sequential_layer_call_and_return_conditional_losses_163862q !@’=
6’3
)&
hidden_1_input?????????η
p

 
ͺ "%’"

0?????????
 »
F__inference_sequential_layer_call_and_return_conditional_losses_163883q !@’=
6’3
)&
hidden_1_input?????????η
p 

 
ͺ "%’"

0?????????
 ³
F__inference_sequential_layer_call_and_return_conditional_losses_164027i !8’5
.’+
!
inputs?????????η
p

 
ͺ "%’"

0?????????
 ³
F__inference_sequential_layer_call_and_return_conditional_losses_164053i !8’5
.’+
!
inputs?????????η
p 

 
ͺ "%’"

0?????????
 
+__inference_sequential_layer_call_fn_163922d !@’=
6’3
)&
hidden_1_input?????????η
p

 
ͺ "?????????
+__inference_sequential_layer_call_fn_163960d !@’=
6’3
)&
hidden_1_input?????????η
p 

 
ͺ "?????????
+__inference_sequential_layer_call_fn_164070\ !8’5
.’+
!
inputs?????????η
p

 
ͺ "?????????
+__inference_sequential_layer_call_fn_164087\ !8’5
.’+
!
inputs?????????η
p 

 
ͺ "??????????
$__inference_signature_wrapper_163987 !J’G
’ 
@ͺ=
;
hidden_1_input)&
hidden_1_input?????????η"/ͺ,
*
output 
output?????????