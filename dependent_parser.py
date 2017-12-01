# This file includes functions for implementing the greedy transition based
# parser with given dependencies in a sentence.

pos_prefix = '<pos> :'
label_prefix = '<label> :'
global_root = '<root>'
global_unknown = '<unknown>'
unknown_prefix = '<unknown> :'
global_null = '<null>'

def left_most(sentence, i):
    '''find current node's left most child'''
    for dependency in sentence:
        if dependency[6] == i:
            return dependency[0]
    return None


def right_most(sentence, i):
    '''find current node's right most child'''
    for dependency in reversed(sentence):
        if dependency[6] == i:
            return dependency[0]
    return None


def search_root(dependencies):
    '''find root of current sentence'''
    for i in range(len(dependencies)):
        if dependencies[i][3] == '0':
            return i


def search_children(dependencies, index):
    '''Search children of a node in a sentence's dependencies'''
    list1 = []
    for i in range(len(dependencies)):
        if dependencies[i][3] == index:
            list1.append(i)
    list1.reverse()
    return list1


def left_arc_most_currently(sentence, i):
    '''find current node's left most child based on current processing result space'''
    if i is not None:
        for dependency in sentence:
            if dependency[3] == i[0] and dependency[0] < i[0]:
                return dependency
    return None


def right_arc_most_currently(sentence, i):
    '''find current node's right most child based on current processing result space'''
    if i is not None:
        for dependency in reversed(sentence):
            if dependency[3] == i[0] and dependency[0] > i[0]:
                return dependency
    return None


def left_arc_most2_currently(sentence, i):
    '''find current node's left most second child based on current processing result space'''
    if i is not None:
        c = 0
        for dependency in sentence:
            if dependency[3] == i[0]:
                c += 1
            if c == 2:
                if dependency[0] < i[0]:
                    return dependency
                break
    return None


def right_arc_most2_currently(sentence, i):
    '''find current node's right most second child based on current processing result space'''
    if i is not None:
        c = 0
        for dependency in reversed(sentence):
            if dependency[3] == i[0]:
                c += 1
            if c == 2:
                if dependency[0] > i[0]:
                    return dependency
                break
    return None


def generate_status_list(stack, buffer, result_dependencies):
    '''generate status_list based on current result of dependency parsing of current sentence
    top 3 elements in stack and top 3 elements in buffer
    left most children of top 2 elements of stack
    right most children of top 2 elements of stack
    left most second children of top 2 elements of stack
    right most second children of top 2 elements of stack
    left most child of left most child of top 2 elements of stack
    right most child of right most child of top 2 elements of stack
    '''
    status_list = [stack[-1] if len(stack) > 0 else None, stack[-2] if len(stack) > 1 else None, stack[-3] if len(stack)>2 else None, buffer[0] if len(buffer) > 0 else None, buffer[1] if len(buffer) > 1 else None, buffer[2] if len(buffer) > 2 else None, ] # top 3 elements in stack and top 3 elements in buffer
    status_list.append(left_arc_most_currently(result_dependencies, status_list[0])) # left child of stack[-1]
    status_list.append(right_arc_most_currently(result_dependencies, status_list[0])) # right child of stack[-1]
    status_list.append(left_arc_most_currently(result_dependencies, status_list[1])) # left child of stack[-2]
    status_list.append(right_arc_most_currently(result_dependencies, status_list[1])) # right child of stack[-2]
    status_list.append(left_arc_most2_currently(result_dependencies, status_list[0])) # left 2nd child of stack[-1]
    status_list.append(right_arc_most2_currently(result_dependencies, status_list[0])) # right 2nd child of stack[-1]
    status_list.append(left_arc_most2_currently(result_dependencies, status_list[1])) # left 2nd child of stack[-2]
    status_list.append(right_arc_most2_currently(result_dependencies, status_list[1])) # right 2nd child of stack[-2]
    status_list.append(left_arc_most_currently(result_dependencies, status_list[6])) # left child of left child of stack[-1]
    status_list.append(right_arc_most_currently(result_dependencies, status_list[7])) # right child of right child of stack[-1]
    status_list.append(left_arc_most_currently(result_dependencies, status_list[8])) # left child of left child of stack[-2]
    status_list.append(right_arc_most_currently(result_dependencies, status_list[9])) # right child of right child of stack[-2]
    return status_list


def extract_features_from_status_list(status_list, token2id):
    '''extract features from a given status list'''
    w_features = [] # word features
    p_features = [] # pos tag features
    l_features = [] # label features
    for c,e in enumerate(status_list):
        if e is not None:
            # current target exists

            # index for word or unknown word
            w_features.append(token2id[e[1]] if e[1] in token2id else token2id[global_unknown])
            # index for pos tag or unknown pos tag, len(e)==5 current element from result_dependencies
            # else current element from stack or buffer (len is 8)
            # in different situation the pos tag in different place in list
            p_features.append(token2id[pos_prefix + e[2]] if (pos_prefix + e[2]) in token2id else token2id[pos_prefix + global_unknown] ) if len(e) == 5 else p_features.append(token2id[pos_prefix + e[4]] if (pos_prefix + e[4]) in token2id else token2id[pos_prefix + global_unknown])
            if c > 5:
                # only when current element is from result_dependencies
                l_features.append(token2id[label_prefix + e[4]] if (label_prefix + e[4]) in token2id else token2id[label_prefix + global_null]) if len(e) == 5 else l_features.append(token2id[label_prefix + e[7]] if (label_prefix + e[7]) in token2id else token2id[label_prefix + global_null]) # label's index
        else:
            # current target not exists
            w_features.append(token2id[global_null]) # null word index
            p_features.append(token2id[pos_prefix + global_null]) # null word's pos tag index
            if c > 5:
                # only when current element is from result_dependencies
                l_features.append(token2id[label_prefix + global_null])
    features = w_features + p_features + l_features # concat three kinds of features
    # print(features)
    return features


def extract_features_from_train_data(dependencies, token2id):
    '''simulate the process of dependency parsing a given sentence
    Extract useful features during this process'''
    stack = [] # initial state, stack is empty
    buffer = dependencies[:] # initial state, buffer has every element of given sentence
    len_total = len(buffer) # number of word tokens in current sentence
    features_list = [] # initialize features_list, this list only for debug
    status_lists = []
    result_dependencies = [[e[0], e[1], None, None, None] for e in buffer] # for storing dependencies which have already been parsed out. Order and Word info initialized here
    right_most_child = dict() # initialize the dict for store every ele's right_most_child_index
    for i in [e[0] for e in dependencies]:
        tmp = right_most(dependencies, i) # get every node's right most child, prevent pop an element too early which would make some children of it be orphans
        right_most_child[i] = int(tmp) if tmp is not None else -1 # set value
    arcs = [] # list for storing arcs
    while len(buffer) > 0 or len(stack) > 1:
        # when buffer is not empty or stack has more than one element
        '''construct features for current status'''
        status_list = generate_status_list(stack, buffer, result_dependencies) # generate all words for extracing features from
        status_lists.append(status_list)
        features = extract_features_from_status_list(status_list, token2id)
        if len(stack) > 1:
            if stack[-2][6] == stack[-1][0] : # left-arc stack[-2]'s parent is stack[-1]
                # store info for debugging
                features_list.append({'features':features, 'op':'left-arc', 'label':label_prefix + stack[-2][7]})
                # add this operation's arc type
                arcs.append('left-arc : ' + stack[-2][7])
                result_dependencies[int(stack[-2][0])-1][2] = stack[-2][4] # pos tag
                result_dependencies[int(stack[-2][0])-1][3] = stack[-2][6] # parent index
                result_dependencies[int(stack[-2][0])-1][4] = stack[-2][7] # label
                stack.pop(-2) # remove the dependant from current stack
            elif stack[-1][6] == stack[-2][0] and len_total-right_most_child[stack[-1][0]]>=len(buffer): # right-arc stack[-1]'s parent is stack[-2], only when stack[-1] does not have more potential right children
                features_list.append({'features':features, 'op':'right-arc', 'label':label_prefix + stack[-1][7]})
                arcs.append('right-arc : ' + stack[-1][7])
                result_dependencies[int(stack[-1][0])-1][2] = stack[-1][4] # pos tag
                result_dependencies[int(stack[-1][0])-1][3] = stack[-1][6] # parent index
                result_dependencies[int(stack[-1][0])-1][4] = stack[-1][7] # label
                stack.pop(-1) # remove the dependant from current stack
            else:
                if len(buffer) == 0:
                    # if buffer is empty, finish
                    break
                # else: this is a shift arc operation
                features_list.append({'features':features, 'op':'shift', 'label':label_prefix + global_null})
                arcs.append('shift') # add shift operation to arcs
                tmp = buffer.pop(0) # pop the top element from buffer
                stack.append(tmp) # push this element into stack
        else:
            # if stack has less than or equal to 1 element, need to pull more element from buffer
            if len(buffer) == 0:
                # if buffer is empty
                break
            features_list.append({'features':features, 'op':'shift', 'label':label_prefix + global_null})
            arcs.append('shift')
            tmp = buffer.pop(0)
            stack.append(tmp)
    # out of the while loop, buffer is empty and stack does not have more than one element
    # only the root node left
    # another possible situation is current sentence cannot be processed with this parser
    # when there are two cross dependencies
    status_list = generate_status_list(stack, buffer, result_dependencies) # generate all words for extracing features from
    features = extract_features_from_status_list(status_list, token2id)
    status_lists.append(status_list)

    if len(stack) > 1:
        # if this is a not be able to handle scenario, return unfinished features_list
        return features_list, status_lists
    # else: root node
    features_list.append({'features':features, 'op':'right-arc', 'label':label_prefix + stack[-1][7]})
    arcs.append('right-arc : ' + stack[-1][7]) # root -> current root node
    result_dependencies[int(stack[-1][0])-1][2] = stack[-1][4] # pos tag
    result_dependencies[int(stack[-1][0])-1][3] = stack[-1][6] # parent (actually zero at here)
    result_dependencies[int(stack[-1][0])-1][4] = stack[-1][7] # label
    stack.pop(-1)
    return features_list, status_lists


class Teststc(object):
    idx = None
    ori_stc = None
    stk = []
    buf = []
    rst = []

    def __init__(self, stc, idx):
        self.idx = idx
        self.ori_stc = stc[:]
        self.stk.append(stc[0])
        self.buf = self.buf + stc[1:len(stc)]
        self.rst = []
        for i in range(len(self.ori_stc)):
            e=self.ori_stc[i]
            self.rst.append([e[0], e[1], e[4], None, None])

    def trans(self, op):
        if len(self.stk) ==0 and len(self.buf) == 0:
            return -1
        if op == 'shift:' + global_null:
            if len(self.buf) != 0:
                self.stk.append(self.buf.pop(0))
                return 0
            else:
                if len(self.stk) == 1:
                    op = 'right-arc:' + 'root'
                else:
                    op = 'left-arc:' + global_null
        if len(self.stk) < 2 and len(self.buf) > 0:
            self.stk.append(self.buf.pop(0))
            return 0
        if len(self.stk) == 1 and len(self.buf) == 0:
            self.rst[int(self.stk[-1][0])-1][3] = '0'
            self.rst[int(self.stk[-1][0])-1][4] = 'root'
            self.stk.pop(-1)
            return 0
        pre_arc = op.split(':')
        if pre_arc[0] == 'left-arc':
            tmp = self.stk.pop(-2)
            self.rst[int(tmp[0])-1][3] = self.stk[-1][0]
            self.rst[int(tmp[0])-1][4] = pre_arc[1]
            return 0
        tmp = self.stk.pop(-1)
        self.rst[int(tmp[0])-1][3] = self.stk[-1][0]
        self.rst[int(tmp[0])-1][4] = pre_arc[1]
        return 0

    def features(self, token2id):
        if len(self.stk) ==0 and len(self.buf) == 0:
            return [0]*48
        return extract_features_from_status_list(generate_status_list(self.stk, self.buf, self.rst), token2id)


def process_test_data(test_data):
    r = []
    for e in test_data:
        r.append(Teststc(e,len(r)))
    return r


def result2graph(dependencies):
    '''Use graphviz to draw parsing tree'''
    dot = Digraph(comment="Parsing tree")
    root = search_root(dependencies)
    dot.node(dependencies[root][0] , dependencies[root][1])
    node_queue = search_children(dependencies, dependencies[root][0])
    while len(node_queue) > 0:
        current_node_index = node_queue.pop(-1)
        current_node = dependencies[current_node_index]
        node_queue += search_children(dependencies, current_node[0])
        dot.node(current_node[0], current_node[1])
        dot.edge(current_node[3], current_node[0], label=current_node[4], dir='back')
    dot.render('./parsing_tree.gv', view=True)


def transition_process_one_sentence(dependencies):
    stack = []
    buffer = dependencies
    len_total = len(buffer)
    result_dependencies = [[str(i+1), e[1], None, None, None] for i, e in enumerate(dependencies)]
    right_most_child = dict() # initialize the dict for store every ele's right_most_child_index
    for i in [e[0] for e in dependencies]:
        tmp = right_most(dependencies, i)
        right_most_child[i] = int(tmp) if tmp is not None else -1
    arcs = []
    while len(buffer) or len(stack) > 1:
        status="stack:[ROOT, {}] buffer:[{}] arc:[".format(', '.join([j[1] for j in stack]), ', '.join([j[1] for j in buffer]))
        if len(stack) > 1:
            if stack[-2][6] == stack[-1][0]: # left-arc
                arcs.append('left-arc : ' + stack[-2][7])
                result_dependencies[int(stack[-2][0])-1][2] = stack[-2][4]
                result_dependencies[int(stack[-2][0])-1][3] = stack[-2][6]
                result_dependencies[int(stack[-2][0])-1][4] = stack[-2][7]
                stack.pop(-2)
            elif stack[-1][6] == stack[-2][0] and len_total-right_most_child[stack[-1][0]]>=len(buffer): # right-arc
                arcs.append('right-arc : ' + stack[-1][7])
                result_dependencies[int(stack[-1][0])-1][2] = stack[-1][4]
                result_dependencies[int(stack[-1][0])-1][3] = stack[-1][6]
                result_dependencies[int(stack[-1][0])-1][4] = stack[-1][7]
                stack.pop(-1)
            else:
                arcs.append('shift')
                tmp = buffer.pop(0)
                stack.append(tmp)
        else:
            arcs.append('shift')
            tmp = buffer.pop(0)
            stack.append(tmp)
#        print(status, arcs[-1], ']') # for debug
    status="stack:[ROOT, {}] buffer:[{}] arc:[".format(', '.join([j[1] for j in stack]), ', '.join([j[1] for j in buffer]))
    arcs.append('right-arc : ' + stack[-1][7])
    result_dependencies[int(stack[-1][0])-1][2] = stack[-1][4]
    result_dependencies[int(stack[-1][0])-1][3] = stack[-1][6]
    result_dependencies[int(stack[-1][0])-1][4] = stack[-1][7]
    stack.pop(-1)
#    print(status, arcs[-1], ']') # for debug
#    result2graph(result_dependencies) # generate graph
