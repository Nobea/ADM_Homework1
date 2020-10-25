#!/usr/bin/env python
# coding: utf-8

# ### Introduction

# Solve Me First


def solveMeFirst(a,b):
    return a+b


num1 = int(input())
num2 = int(input())
res = solveMeFirst(num1,num2)
print(res)


# Say "Hello, World!" With Python



print("Hello, World!")


# Python If-Else
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n % 2 != 0 :
        print("Weird")
    elif n <= 5:
        print("Not Weird")
    elif n <= 20:
        print("Weird")
    else:
        print("Not Weird")


# Arithmetic Operators


if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)


# Python: Division



if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)


# Write a function


def is_leap(year):
    leap = False
    if year % 400 == 0 and year % 100 != 0:
        leap = True
    elif year % 4 == 0:
        leap = True
    return leap


# Loops


if __name__ == '__main__':
    n = int(input())
    for i in range (n):
        print(i**2)


# ### Data Types

# List Comprehensions



if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

output = [[i,j,k] for i in range(x+1) for j in range(y+1) for k in range (z+1) if i+j+k != n]
print (output)


# Find the Runner-Up Score!



if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())

    print (sorted(set(arr))[-2])


# Nested Lists


if __name__ == '__main__':
    students = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        students.append([name, score])
    
    scores = sorted(set([i[1] for i in students]))
    second_lowest_score = scores[1]

    names_second_lowest = sorted([i[0] for i in students if i[1] == second_lowest_score])

    for i in names_second_lowest:
        print(i)


# Finding the percentage


if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

    query_score = student_marks[query_name]
    
    print("{0:.2f}".format(sum(query_score)/len(query_score)))


# Lists


if __name__ == '__main__':
    N = int(input())
    lis = []
    for _ in range(N):
        line = input().split()
        command = line[0]
        arguments = line[1:]
        if command == "print":
            print(lis)
        else:
            command += "(" + ",".join(arguments) + ")"
            eval("lis." + command)


# Tuples



if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    
    print(hash(tuple(integer_list)))


# ### Strings

# sWAP cASE


def swap_case(s):
    return s.swapcase()


# String Split and Join


def split_and_join(line):
    return "-".join(line.split(" "))


# What's Your Name?


def print_full_name(a, b):
    print("Hello {} {}! You just delved into python.".format(a, b))


# Mutations


def mutate_string(string, position, character):
    return string[:position] + character + string[position+1:]


# Find a string


def count_substring(string, sub_string):
    counts = 0
    sub_len = len(sub_string)
    for i in range(len(string)):
        if string[i:i+sub_len] == sub_string:
            counts += 1
    return counts


# String Validators


if __name__ == '__main__':
    s = input()
    print (any(c.isalnum()  for c in s))
    print (any(c.isalpha() for c in s))
    print (any(c.isdigit() for c in s))
    print (any(c.islower() for c in s))
    print (any(c.isupper() for c in s))


# Text Wrap


def wrap(string, max_width):
    return textwrap.fill(string, width = max_width)


# String Formatting


def print_formatted(number):
    width = len("{0:b}".format(n))
    for i in range(1,n+1):
        print("{0:{width}d} {0:{width}o} {0:{width}X} {0:{width}b}".format(i, width=width))


# Capitalize!



def solve(s):
    words = s.split()
    cap_words = [word.capitalize() for word in words]
    return " ".join(cap_words)


# ### Sets

# Introduction to Sets


def average(array):
    types = set(array)
    num = len(types)
    return sum(types)/num


# Symmetric Difference



M = int(input())
s1 = set(map(int, input().split()))
N = int(input())
s2 = set(map(int, input().split()))

result = sorted(list(s1.symmetric_difference(s2)))

for i in result:
    print(i)


# Set .add()



N = int(input())
s = set()

for i in range(N):
    s.add(input())

print(len(s))


# Set .union() Operation



n = int(input())
s1 = set(map(int, input().split()))
b = int(input())
s2 = set(map(int, input().split()))

print(len(s1.union(s2)))


# Set .intersection() Operation



n = int(input())
s1 = set(map(int, input().split()))
b = int(input())
s2 = set(map(int, input().split()))

print(len(s1.intersection(s2)))


# Set .difference() Operation



n = int(input())
s1 = set(map(int, input().split()))
b = int(input())
s2 = set(map(int, input().split()))

print(len(s1.difference(s2)))


# Set .symmetric_difference() Operation


n = int(input())
s1 = set(map(int, input().split()))
b = int(input())
s2 = set(map(int, input().split()))

print(len(s1.symmetric_difference(s2)))


# The Captain's Room


K = int(input())
room_num = list(map(int, input().split()))
single_room_num = set(room_num)

for i in single_room_num:
    room_num.remove(i)

no_cap = set(room_num)
single_room_num.difference_update(no_cap)

print("".join(str(e) for e in single_room_num))


# Check Strict Superset


A = set(map(int, input().split()))
n = int(input())
TF = []
for i in range(n):
    s = set(map(int, input().split()))
    TF.append(A.issuperset(s))

if False in TF:
    print("False")


# No Idea!


n, m = map(int, input().split())
arr = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))

happiness = 0 

for i in arr:
    if i in A:
        happiness += 1
    if i in B:
        happiness -= 1

print(happiness)


# ### Collections

# collections.Counter()


from collections import Counter

X = int(input())
shoe_sizes = Counter(list(map(int, input().split())))

N = int(input())
prices = {}

for i in range(N):
    size, price = map(int, input().split())
    prices.setdefault(size, []).append(price)

earning = 0

for key in prices:
    if key in shoe_sizes.keys():
        earning += sum(prices[key][:shoe_sizes[key]])

print(earning)


# DefaultDict Tutorial


from collections import defaultdict

n, m = map(int, input().split())

A = []
B = []

for i in range(n):
    A.append(input())

for e in range(m):
    B.append(input())

occurrences = {}
occurrences = defaultdict(list)

for word in B:
    occurrences[word] = [i+1 for i, j in enumerate(A) if j == word]
    if len(occurrences[word]) > 0:
        print(" ".join(map(str, occurrences[word])))
    else:
        print(-1)


# Collections.namedtuple()


from collections import namedtuple
N = int(input())
student = namedtuple("student", input().split())

mark_sum = 0

for i in range(N):
    a, b, c, d = input().split()
    stud = student(a, b, c, d)
    mark_sum += int(stud.MARKS)

print(mark_sum/N)


# Collections.OrderedDict()


from collections import OrderedDict
from collections import defaultdict

N = int(input())

sales = OrderedDict()
sales = defaultdict(int)

for i in range(N):
    line = input().split()
    net_price = int(line[-1])
    line.remove(line[-1])
    item_name = " ".join(line)
    sales[item_name] += net_price

for i in sales:
    print(i, sales[i])


# Word Order


from collections import OrderedDict
from collections import defaultdict

n = int(input())

words = OrderedDict()
words = defaultdict(int)

for i in range(n):
    word = input()
    words[word] += 1

print(len(words))
print(" ".join(map(str, [words[i] for i in words])))


# ### Date and Time

# Calendar Module


import calendar
m, d, y = map(int, input().split())
week = ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
day = calendar.weekday(y, m, d)
print(week[day])


# ### Numpy

# Arrays


def arrays(arr):
    n_arr = numpy.array(arr[::-1], float)
    return n_arr


# Shape and Reshape


import numpy as np

arr = np.array(list(map(int, input().split())))

arr = np.reshape(arr, (3,3))

print(arr)


# Transpose and Flatten



import numpy as np

N, M = map(int, input().split())

arr = []

for i in range(N):
    arr.append(list(map(int, input().split())))
    
arr = np.array(arr)

print(np.transpose(arr))
print(arr.flatten())


# Concatenate


import numpy as np

N, M, P = map(int, input().split())

arr = []

for i in range(N):
    arr.append(list(map(int, input().split())))

for i in range(M):
    arr.append(list(map(int, input().split())))


print(np.array(arr))


# Zeros and Ones



import numpy as np

dim = tuple(map(int, input().split()))

print(np.zeros(dim, dtype = np.int))
print(np.ones(dim, dtype = np.int))


# Eye and Identity


import numpy as np
print(str(np.eye(*map(int,input().split()))).replace('1',' 1').replace('0',' 0'))


# Array Mathematics


import numpy as np

N, M = map(int, input().split())
A = []
B = []

for i in range(N):
    A.append(list(map(int, input().split())))

for i in range(N):
    B.append(list(map(int, input().split())))

A = np.array(A)
B = np.array(B)

print(A+B)
print(A-B)
print(A*B)
print(A//B)
print(A%B)
print(A**B)


# Floor, Ceil and Rint


import numpy as np

np.set_printoptions(sign=' ')

a = np.array(input().split(),float)

print(np.floor(a))
print(np.ceil(a))
print(np.rint(a))


# Sum and Prod


import numpy as np

N, M = map(int, input().split())

arr = []

for i in range(N):
    arr.append(list(map(int, input().split())))

arr = np.array(arr)

print(np.prod(np.sum(arr, axis = 0)))


# Min and Max


import numpy as np

N, M = map(int, input().split())

arr = []

for i in range(N):
    arr.append(list(map(int, input().split())))

arr = np.array(arr)

print(np.max(np.min(arr, axis = 1)))


# Mean, Var and Std


import numpy as np

np.set_printoptions(legacy='1.13')

N, M = map(int, input().split())

arr = []

for i in range(N):
    arr.append(list(map(int, input().split())))

arr = np.array(arr)

print(np.mean(arr, axis = 1))
print(np.var(arr, axis = 0))
print(np.std(arr))


# Dot and Cross


import numpy as np

N = int(input())

A = []
B = []

for i in range(N):
    A.append(list(map(int, input().split())))

for i in range(N):
    B.append(list(map(int, input().split())))

A = np.array(A)
B = np.array(B)

print(np.matmul(A, B))


# Inner and Outer


import numpy as np

A = np.array(list(map(int, input().split())))
B = np.array(list(map(int, input().split())))

print(np.inner(A, B))
print(np.outer(A, B))


# Polynomials


import numpy as np

P = list(map(float, np.array(input().split())))
x = int(input())

print(np.polyval(P, x))


# Linear Algebra


import numpy as np

N = int(input())

A = []

for i in range(N):
    A.append(list(map(float, input().split())))

A = np.array(A)

print(round(np.linalg.det(A), 2))


# ### Exceptions


T = int(input())

for i in range(T):
    try:
        a, b = map(int, input().split())
        print(a//b)
    except (ZeroDivisionError, ValueError) as e:
        print("Error Code:", e)


# ### Built-ins

# Zipped!



N, X = map(int, input().split())
marks = []
for i in range(X):
    marks.append(list(map(float, input().split())))

for i in zip(*marks):
    print(round(sum(i)/X, 1))


# ### Python Functionals

# Map and Lambda Function


cube = lambda x: x**3

def fibonacci(n):
    fib = [0, 1]
    if n == 1 or n ==2:
        return fib[:n]
    if n > 2:
        fib = fibonacci(n-1)
        fib.append(fibonacci(n-1)[-1]+fibonacci(n-2)[-1])
        return fib


# ### Regex and Parsing challenges

# Detect Floating Point Number


import re 

T = int(input())

for i in range(T):
    x = input()
    print(bool(re.match(r"^[-+]?[0-9]*\.[0-9]+$", x)))


# Re.split()


regex_pattern = r"[,.]+"


# Re.findall() & Re.finditer()



import re

s = input()

m = re.findall(r'(?<=[^aeiuo])([aeiuo]{2,})(?=[^aeiuo])', s, flags=re.I)
print('\n'.join(m or ['-1']))


# ### Python Challenges

# Birthday Cake Candles


import math
import os
import random
import re
import sys
from collections import Counter

def birthdayCakeCandles(candles):
    max_height = max(candles)
    height_count = Counter(candles)
    return height_count[max_height]

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()


# Number Line Jumps


import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    if v1 > v2 and x1 > x2:
        return "NO"
    elif v2 > v1 and x2 > x1:
        return "NO"
    elif v2-v1 != 0 and (x1-x2)%(v2-v1) == 0:
        return "YES"
    else:
        return "NO"
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    x1V1X2V2 = input().split()
    x1 = int(x1V1X2V2[0])
    v1 = int(x1V1X2V2[1])
    x2 = int(x1V1X2V2[2])
    v2 = int(x1V1X2V2[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()


# Recursive Digit Sum


import math
import os
import random
import re
import sys


def superDigit(n, k):
    num = int(n*k)
    while len(str(num)) > 1:
        num = sum(map(int, str(num)))
    return num

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    nk = input().split()
    n = nk[0]
    k = int(nk[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()


# Insertion Sort - Part 1


import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    last_digit = arr[-1]
    for i in range(n-2, -1, -1):
        if arr[i] >= last_digit:
            arr[i+1] = arr[i]
            print(" ".join(map(str, arr)))
        else:
            arr[i+1] = last_digit
            print(" ".join(map(str, arr)))
            break
    if last_digit not in arr:
        arr[0] = last_digit
        print(" ".join(map(str, arr)))


if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)


