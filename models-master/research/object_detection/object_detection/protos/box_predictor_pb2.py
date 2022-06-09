# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/box_predictor.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from object_detection.protos import hyperparams_pb2 as object__detection_dot_protos_dot_hyperparams__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n+object_detection/protos/box_predictor.proto\x12\x17object_detection.protos\x1a)object_detection/protos/hyperparams.proto\"\x90\x03\n\x0c\x42oxPredictor\x12Y\n\x1b\x63onvolutional_box_predictor\x18\x01 \x01(\x0b\x32\x32.object_detection.protos.ConvolutionalBoxPredictorH\x00\x12P\n\x17mask_rcnn_box_predictor\x18\x02 \x01(\x0b\x32-.object_detection.protos.MaskRCNNBoxPredictorH\x00\x12G\n\x12rfcn_box_predictor\x18\x03 \x01(\x0b\x32).object_detection.protos.RfcnBoxPredictorH\x00\x12s\n)weight_shared_convolutional_box_predictor\x18\x04 \x01(\x0b\x32>.object_detection.protos.WeightSharedConvolutionalBoxPredictorH\x00\x42\x15\n\x13\x62ox_predictor_oneof\"\xaf\x04\n\x19\x43onvolutionalBoxPredictor\x12>\n\x10\x63onv_hyperparams\x18\x01 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12\x14\n\tmin_depth\x18\x02 \x01(\x05:\x01\x30\x12\x14\n\tmax_depth\x18\x03 \x01(\x05:\x01\x30\x12&\n\x1bnum_layers_before_predictor\x18\x04 \x01(\x05:\x01\x30\x12\x19\n\x0buse_dropout\x18\x05 \x01(\x08:\x04true\x12%\n\x18\x64ropout_keep_probability\x18\x06 \x01(\x02:\x03\x30.8\x12\x16\n\x0bkernel_size\x18\x07 \x01(\x05:\x01\x31\x12\x18\n\rbox_code_size\x18\x08 \x01(\x05:\x01\x34\x12&\n\x17\x61pply_sigmoid_to_scores\x18\t \x01(\x08:\x05\x66\x61lse\x12%\n\x1a\x63lass_prediction_bias_init\x18\n \x01(\x02:\x01\x30\x12\x1c\n\ruse_depthwise\x18\x0b \x01(\x08:\x05\x66\x61lse\x12j\n\x18\x62ox_encodings_clip_range\x18\x0c \x01(\x0b\x32H.object_detection.protos.ConvolutionalBoxPredictor.BoxEncodingsClipRange\x1a\x31\n\x15\x42oxEncodingsClipRange\x12\x0b\n\x03min\x18\x01 \x01(\x02\x12\x0b\n\x03max\x18\x02 \x01(\x02\"\xad\x06\n%WeightSharedConvolutionalBoxPredictor\x12>\n\x10\x63onv_hyperparams\x18\x01 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12.\n\x1f\x61pply_conv_hyperparams_to_heads\x18\x13 \x01(\x08:\x05\x66\x61lse\x12/\n apply_conv_hyperparams_pointwise\x18\x14 \x01(\x08:\x05\x66\x61lse\x12&\n\x1bnum_layers_before_predictor\x18\x04 \x01(\x05:\x01\x30\x12\x10\n\x05\x64\x65pth\x18\x02 \x01(\x05:\x01\x30\x12\x16\n\x0bkernel_size\x18\x07 \x01(\x05:\x01\x33\x12\x18\n\rbox_code_size\x18\x08 \x01(\x05:\x01\x34\x12%\n\x1a\x63lass_prediction_bias_init\x18\n \x01(\x02:\x01\x30\x12\x1a\n\x0buse_dropout\x18\x0b \x01(\x08:\x05\x66\x61lse\x12%\n\x18\x64ropout_keep_probability\x18\x0c \x01(\x02:\x03\x30.8\x12%\n\x16share_prediction_tower\x18\r \x01(\x08:\x05\x66\x61lse\x12\x1c\n\ruse_depthwise\x18\x0e \x01(\x08:\x05\x66\x61lse\x12p\n\x0fscore_converter\x18\x10 \x01(\x0e\x32M.object_detection.protos.WeightSharedConvolutionalBoxPredictor.ScoreConverter:\x08IDENTITY\x12v\n\x18\x62ox_encodings_clip_range\x18\x11 \x01(\x0b\x32T.object_detection.protos.WeightSharedConvolutionalBoxPredictor.BoxEncodingsClipRange\x1a\x31\n\x15\x42oxEncodingsClipRange\x12\x0b\n\x03min\x18\x01 \x01(\x02\x12\x0b\n\x03max\x18\x02 \x01(\x02\"+\n\x0eScoreConverter\x12\x0c\n\x08IDENTITY\x10\x00\x12\x0b\n\x07SIGMOID\x10\x01\"\xbf\x04\n\x14MaskRCNNBoxPredictor\x12<\n\x0e\x66\x63_hyperparams\x18\x01 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12\x1a\n\x0buse_dropout\x18\x02 \x01(\x08:\x05\x66\x61lse\x12%\n\x18\x64ropout_keep_probability\x18\x03 \x01(\x02:\x03\x30.5\x12\x18\n\rbox_code_size\x18\x04 \x01(\x05:\x01\x34\x12>\n\x10\x63onv_hyperparams\x18\x05 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12%\n\x16predict_instance_masks\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\'\n\x1amask_prediction_conv_depth\x18\x07 \x01(\x05:\x03\x32\x35\x36\x12 \n\x11predict_keypoints\x18\x08 \x01(\x08:\x05\x66\x61lse\x12\x17\n\x0bmask_height\x18\t \x01(\x05:\x02\x31\x35\x12\x16\n\nmask_width\x18\n \x01(\x05:\x02\x31\x35\x12*\n\x1fmask_prediction_num_conv_layers\x18\x0b \x01(\x05:\x01\x32\x12\'\n\x18masks_are_class_agnostic\x18\x0c \x01(\x08:\x05\x66\x61lse\x12\'\n\x18share_box_across_classes\x18\r \x01(\x08:\x05\x66\x61lse\x12+\n\x1c\x63onvolve_then_upsample_masks\x18\x0e \x01(\x08:\x05\x66\x61lse\"\xf9\x01\n\x10RfcnBoxPredictor\x12>\n\x10\x63onv_hyperparams\x18\x01 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12\"\n\x17num_spatial_bins_height\x18\x02 \x01(\x05:\x01\x33\x12!\n\x16num_spatial_bins_width\x18\x03 \x01(\x05:\x01\x33\x12\x13\n\x05\x64\x65pth\x18\x04 \x01(\x05:\x04\x31\x30\x32\x34\x12\x18\n\rbox_code_size\x18\x05 \x01(\x05:\x01\x34\x12\x17\n\x0b\x63rop_height\x18\x06 \x01(\x05:\x02\x31\x32\x12\x16\n\ncrop_width\x18\x07 \x01(\x05:\x02\x31\x32')



_BOXPREDICTOR = DESCRIPTOR.message_types_by_name['BoxPredictor']
_CONVOLUTIONALBOXPREDICTOR = DESCRIPTOR.message_types_by_name['ConvolutionalBoxPredictor']
_CONVOLUTIONALBOXPREDICTOR_BOXENCODINGSCLIPRANGE = _CONVOLUTIONALBOXPREDICTOR.nested_types_by_name['BoxEncodingsClipRange']
_WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR = DESCRIPTOR.message_types_by_name['WeightSharedConvolutionalBoxPredictor']
_WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR_BOXENCODINGSCLIPRANGE = _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR.nested_types_by_name['BoxEncodingsClipRange']
_MASKRCNNBOXPREDICTOR = DESCRIPTOR.message_types_by_name['MaskRCNNBoxPredictor']
_RFCNBOXPREDICTOR = DESCRIPTOR.message_types_by_name['RfcnBoxPredictor']
_WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR_SCORECONVERTER = _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR.enum_types_by_name['ScoreConverter']
BoxPredictor = _reflection.GeneratedProtocolMessageType('BoxPredictor', (_message.Message,), {
  'DESCRIPTOR' : _BOXPREDICTOR,
  '__module__' : 'object_detection.protos.box_predictor_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.BoxPredictor)
  })
_sym_db.RegisterMessage(BoxPredictor)

ConvolutionalBoxPredictor = _reflection.GeneratedProtocolMessageType('ConvolutionalBoxPredictor', (_message.Message,), {

  'BoxEncodingsClipRange' : _reflection.GeneratedProtocolMessageType('BoxEncodingsClipRange', (_message.Message,), {
    'DESCRIPTOR' : _CONVOLUTIONALBOXPREDICTOR_BOXENCODINGSCLIPRANGE,
    '__module__' : 'object_detection.protos.box_predictor_pb2'
    # @@protoc_insertion_point(class_scope:object_detection.protos.ConvolutionalBoxPredictor.BoxEncodingsClipRange)
    })
  ,
  'DESCRIPTOR' : _CONVOLUTIONALBOXPREDICTOR,
  '__module__' : 'object_detection.protos.box_predictor_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.ConvolutionalBoxPredictor)
  })
_sym_db.RegisterMessage(ConvolutionalBoxPredictor)
_sym_db.RegisterMessage(ConvolutionalBoxPredictor.BoxEncodingsClipRange)

WeightSharedConvolutionalBoxPredictor = _reflection.GeneratedProtocolMessageType('WeightSharedConvolutionalBoxPredictor', (_message.Message,), {

  'BoxEncodingsClipRange' : _reflection.GeneratedProtocolMessageType('BoxEncodingsClipRange', (_message.Message,), {
    'DESCRIPTOR' : _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR_BOXENCODINGSCLIPRANGE,
    '__module__' : 'object_detection.protos.box_predictor_pb2'
    # @@protoc_insertion_point(class_scope:object_detection.protos.WeightSharedConvolutionalBoxPredictor.BoxEncodingsClipRange)
    })
  ,
  'DESCRIPTOR' : _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR,
  '__module__' : 'object_detection.protos.box_predictor_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.WeightSharedConvolutionalBoxPredictor)
  })
_sym_db.RegisterMessage(WeightSharedConvolutionalBoxPredictor)
_sym_db.RegisterMessage(WeightSharedConvolutionalBoxPredictor.BoxEncodingsClipRange)

MaskRCNNBoxPredictor = _reflection.GeneratedProtocolMessageType('MaskRCNNBoxPredictor', (_message.Message,), {
  'DESCRIPTOR' : _MASKRCNNBOXPREDICTOR,
  '__module__' : 'object_detection.protos.box_predictor_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.MaskRCNNBoxPredictor)
  })
_sym_db.RegisterMessage(MaskRCNNBoxPredictor)

RfcnBoxPredictor = _reflection.GeneratedProtocolMessageType('RfcnBoxPredictor', (_message.Message,), {
  'DESCRIPTOR' : _RFCNBOXPREDICTOR,
  '__module__' : 'object_detection.protos.box_predictor_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.RfcnBoxPredictor)
  })
_sym_db.RegisterMessage(RfcnBoxPredictor)

if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _BOXPREDICTOR._serialized_start=116
  _BOXPREDICTOR._serialized_end=516
  _CONVOLUTIONALBOXPREDICTOR._serialized_start=519
  _CONVOLUTIONALBOXPREDICTOR._serialized_end=1078
  _CONVOLUTIONALBOXPREDICTOR_BOXENCODINGSCLIPRANGE._serialized_start=1029
  _CONVOLUTIONALBOXPREDICTOR_BOXENCODINGSCLIPRANGE._serialized_end=1078
  _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR._serialized_start=1081
  _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR._serialized_end=1894
  _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR_BOXENCODINGSCLIPRANGE._serialized_start=1029
  _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR_BOXENCODINGSCLIPRANGE._serialized_end=1078
  _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR_SCORECONVERTER._serialized_start=1851
  _WEIGHTSHAREDCONVOLUTIONALBOXPREDICTOR_SCORECONVERTER._serialized_end=1894
  _MASKRCNNBOXPREDICTOR._serialized_start=1897
  _MASKRCNNBOXPREDICTOR._serialized_end=2472
  _RFCNBOXPREDICTOR._serialized_start=2475
  _RFCNBOXPREDICTOR._serialized_end=2724
# @@protoc_insertion_point(module_scope)
