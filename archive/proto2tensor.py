import protobuf.CMsgBotWorldState_pb2 as _pb
import google.protobuf.text_format as txtf
from collections import OrderedDict
import numpy as np


proto_obj = _pb.CMsgBotWorldState.RuneInfo


def field_names_from_proto(obj, all_names):
    """Get all field names recursively from a proto.

    NOTE: `oneof` not supported.
    """
    for descriptor in obj.DESCRIPTOR.fields:
        if descriptor.full_name in all_names: continue  # Skip messages we've visited.
        all_names.append(descriptor.full_name)
        value = getattr(obj, descriptor.name)
        if descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                value = descriptor.message_type._concrete_class()
            field_names_from_proto(value, all_names)

def dump_object(obj, values):
    """
    TODO:
    """
    for descriptor in obj.DESCRIPTOR.fields:
        value = getattr(obj, descriptor.name)
        if descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                # map(dump_object, value)
                for v in value:
                    dump_object(v, values)
            else:
                dump_object(value, values)
        elif descriptor.type == descriptor.TYPE_ENUM:
            enum_name = descriptor.enum_type.values[value].name
            # print("%s: enum=%s value=%s" % (descriptor.full_name, enum_name, value))
            values.append((descriptor.full_name, value))
        else:
            # print("%s: value=%s" % (descriptor.full_name, value))
            values.append((descriptor.full_name, value))




field_names = []
field_names_from_proto(proto_obj(), field_names)
print('amount of field_names: %s' % len(field_names))
print(field_names)






prototxt_file = '/Users/tzaman/Drive/code/dotabot/resources/dota_ex.prototxt'
with open(prototxt_file, 'r')  as file:
    data = file.read()

data = """
  type: -1
  location {
    x: -1712.0
    y: 1184.0
    z: 176.0
  }
  status: 0
  time_since_seen: 29.926025390625
"""


data_frame = proto_obj()
txtf.Merge(data, data_frame)

values = []
dump_object(data_frame, values)
print('values=%s' % values)


value_index = []
for key, value in values:
    print(key, value)
    value_index.append(field_names.index(key))
# print('x')
# print(value_index)
# print(len(value_index))




