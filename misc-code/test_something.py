# #
# labels_dict = {}
# number_to_label = {}
#
# file = open('labels', 'r')
# numerical_transition = 0
#
# for line in file:
#     label = line.split('\n')[0]
#
#     numerical_transition += 1
#
#     labels_dict[label] = numerical_transition
#     number_to_label[numerical_transition] = label
#
# #ordered = sorted(labels_dict)
# print(labels_dict)
# print(number_to_label)


# t1 = [0, 0, 0, -28, 0, 0, 0, 0, -38, -28, 15, 43, 0, 0, 0, 0, 0, 0, -25, -28, 0, 0, 0, 0, 0, -28, -28, 0, 0, 15, 43, 15, 43, 43, 15, 43, 15, 43, 15, 0, 49, 1]
# t2 = [0, 0, 0, 0, 0, 0, -28, 15, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -28, -28, 0, 0, 0, 0, 0, -25, -28, 15, 43, 43, 15, 43, 15, 43, 43, 43, 43, 15, 43, 0, 49, 1, 43, 1]
# t3 = [0, 0, 0, 0, 0, 0, 0, -28, -28, 0, 0, 0, 0, 0, 0, 0, 0, 0, -28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -28, -28, 0, 0, 0, 0, 0, 0, 0, 0, 0, -28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -28, -28, -28, 0, 0, 0, -28, 15, 43, 15, 43, 43, 43, 43, 0, 49, 1, 43, 1, 43, 43, 43, 1, 43, 15, 43, 15, 43, 1, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 1, 43, 43, 43, 15, 43, 43, 1, 43, 43, 15, 43, 1, 43, 15, 43, 1]
#
# for t in [t1, t2, t3]:
#     print('length:')
#     print(len(t))
#     shift_count = 0
#     for i in t:
#         if i == 0:
#             shift_count += 1
#
#     print('# shifts')
#     print(shift_count)

print(50*[0])