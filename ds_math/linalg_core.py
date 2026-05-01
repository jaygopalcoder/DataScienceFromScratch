"""Write the following functions without using any external libraries:vector_add(v: List[float], w: List[float]) -> List[float]Adds corresponding elements.Requirement: Add an assert to ensure vectors are the same length.vector_sum(vectors: List[List[float]]) -> List[float]Takes a list of vectors and returns a single vector that is the element-wise sum of all of them.Elite Hint: Use the reduce function from functools or a very clean loop.dot_product(v: List[float], w: List[float]) -> floatThe sum of their component-wise products: $v_1w_1 + v_2w_2 + ... + v_nw_n$."""


from typing import List
from functools import reduce
from math import sqrt

def vector_add(v: List[float], w: List[float]) ->List[float]:

    assert len(v) == len(w), "For vector addition both the vectors should be of same length."
    u_v = [item1 + item2 for item1, item2 in zip(v,w)]
    return u_v

def vector_sub(v: List[float], w: List[float]) ->List[float]:

    assert len(v) == len(w), "For vector subtraction both the vectors should be of same length."
    u_v = [item1 - item2 for item1, item2 in zip(v,w)] # don't create a extra list it is not sustainable for memory simply return the generator
    return u_v

def vector_sum(vectors: List[List[float]]) -> List[float]:

    vectors_len = len(vectors[0])
    # my first logic for comparison of length of all vectors
    for vector in vectors:
        assert len(vector) == vectors_len, "All vector should be of same length"

    # this can be also implemented using all
    # assert all(len(vector) == vectors_len for vector in vectors)

    # the logic of sum of corresponding element is simple increment the index and for particular index jump to all another vector and acess their specific position element
    # this is the first logic i have written for sum of corresponding elements however it has time complexity of O(m*n) 
    # return [sum(vector[i] for vector in vectors) for i in range(vectors_len)]
    # we can use reduce function for this purpose which provides time complexity of O(n)
    return reduce(vector_add, vectors)  # it adds the first two list of vectors then add resultant with the next list and so on until it becomes one list

def dot_product(v: List[float], w:List[float]) -> float:
    u_v = [item1*item2 for item1, item2 in zip(v,w)]
    return sum(u_v)

def sum_of_squares(v: List[float]) -> float:
    return dot_product(v,v)

def magnitude(v: List[float]) -> float:
    return sqrt(sum_of_squares(v))

def distance(v: List[float], w: List[float]) -> float:
    return magnitude(vector_sub(v, w))


u = [9,3.9,4,53]
v = [2,2.3,9,2]
print(vector_add(u,v))
print(dot_product(u,v))


"""WHAT IS REDUCE FUNCTION ?
IT TAKES TWO TWO ARGUMENTS ONE IS A FUCNTION WHICH CAN ACCEPTS TWO PARAMETERS AND ITERATIVELY APPLY THE FUNCTION OVER THE LIST,TUPLE ETC UNTIL IT GIVES SINGLE RESULT OR LIST,TUPLE.
EX: """

nums = [92,2,4,2,3]
nums_sum = reduce(lambda x,y : x+y, nums)
print(nums_sum)
# in place of lambda you can pass reference of any function which takes two arguments and reduce will apply that function over the list or tuple until it gives single result

"""Why did we just calculate distance and magnitude?

Loss Functions: In Tesla's object detection, the "loss" (error) is often the distance between the predicted coordinates of a pedestrian and the actual coordinates.

Regularization: To prevent an AI from "overfitting" (memorizing data), we penalize the magnitude of its weight vectors. This is called L2 Regularization. If the weights get too "long" (high magnitude), we force them back down."""



