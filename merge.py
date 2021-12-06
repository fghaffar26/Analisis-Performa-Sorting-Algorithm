# sumber https://www.geeksforgeeks.org/

import timeit
import math

from numpy.random import seed
from numpy.random import randint
import matplotlib.pyplot as plt

def partition(start, end, array):
	
	# Initializing pivot's index to start
	pivot_index = start
	pivot = array[pivot_index]
	
	# This loop runs till start pointer crosses
	# end pointer, and when it does we swap the
	# pivot with element on end pointer
	while start < end:
		
		# Increment the start pointer till it finds an
		# element greater than pivot
		while start < len(array) and array[start] <= pivot:
			start += 1
			
		# Decrement the end pointer till it finds an
		# element less than pivot
		while array[end] > pivot:
			end -= 1
		
		# If start and end have not crossed each other,
		# swap the numbers on start and end
		if(start < end):
			array[start], array[end] = array[end], array[start]
	
	# Swap pivot element with element on end pointer.
	# This puts pivot on its correct sorted place.
	array[end], array[pivot_index] = array[pivot_index], array[end]
	
	# Returning end pointer to divide the array into 2
	return end
	
# The main function that implements QuickSort
def quick_sort(start, end, array):
	
	if (start < end):
		
		# p is partitioning index, array[p]
		# is at right place
		p = partition(start, end, array)
		
		# Sort elements before partition
		# and after partition
		quick_sort(start, p - 1, array)
		quick_sort(p + 1, end, array)

def insertion_sort(lst):
    for i in range(1, len(lst)):
        for j in range(i, 0, -1):
            if lst[j-1] > lst[j]:
                lst[j-1], lst[j] = lst[j], lst[j-1]
            else:
                break

# find left child of node i
def left(i):
	return 2 * i + 1

# find right child of node i
def right(i):
	return 2 * i + 2

# calculate and return array size
def heapSize(A):
	return len(A)-1


# This function takes an array and Heapyfies
# the at node i
def MaxHeapify(A, i):
	# print("in heapy", i)
	l = left(i)
	r = right(i)
	
	# heapSize = len(A)
	# print("left", l, "Rightt", r, "Size", heapSize)
	if l<= heapSize(A) and A[l] > A[i] :
		largest = l
	else:
		largest = i
	if r<= heapSize(A) and A[r] > A[largest]:
		largest = r
	if largest != i:
	# print("Largest", largest)
		A[i], A[largest]= A[largest], A[i]
	# print("List", A)
		MaxHeapify(A, largest)
	
# this function makes a heapified array
def BuildMaxHeap(A):
	for i in range(int(heapSize(A)/2)-1, -1, -1):
		MaxHeapify(A, i)
		
# Sorting is done using heap of array
def HeapSort(A):
	BuildMaxHeap(A)
	B = list()
	heapSize1 = heapSize(A)
	for i in range(heapSize(A), 0, -1):
		A[0], A[i]= A[i], A[0]
		B.append(A[heapSize1])
		A = A[:-1]
		heapSize1 = heapSize1-1
		MaxHeapify(A, 0)

# Python3 program to perform basic timSort
MIN_MERGE = 32

def calcMinRun(n):
	"""Returns the minimum length of a
	run from 23 - 64 so that
	the len(array)/minrun is less than or
	equal to a power of 2.

	e.g. 1=>1, ..., 63=>63, 64=>32, 65=>33,
	..., 127=>64, 128=>32, ...
	"""
	r = 0
	while n >= MIN_MERGE:
		r |= n & 1
		n >>= 1
	return n + r


# This function sorts array from left index to
# to right index which is of size atmost RUN
def insertionSort(arr, left, right):
	for i in range(left + 1, right + 1):
		j = i
		while j > left and arr[j] < arr[j - 1]:
			arr[j], arr[j - 1] = arr[j - 1], arr[j]
			j -= 1


# Merge function merges the sorted runs
def merge(arr, l, m, r):
	
	# original array is broken in two parts
	# left and right array
	len1, len2 = m - l + 1, r - m
	left, right = [], []
	for i in range(0, len1):
		left.append(arr[l + i])
	for i in range(0, len2):
		right.append(arr[m + 1 + i])

	i, j, k = 0, 0, l
	
	# after comparing, we merge those two array
	# in larger sub array
	while i < len1 and j < len2:
		if left[i] <= right[j]:
			arr[k] = left[i]
			i += 1

		else:
			arr[k] = right[j]
			j += 1

		k += 1

	# Copy remaining elements of left, if any
	while i < len1:
		arr[k] = left[i]
		k += 1
		i += 1

	# Copy remaining element of right, if any
	while j < len2:
		arr[k] = right[j]
		k += 1
		j += 1


# Iterative Timsort function to sort the
# array[0...n-1] (similar to merge sort)
def timSort(arr):
	n = len(arr)
	minRun = calcMinRun(n)
	
	# Sort individual subarrays of size RUN
	for start in range(0, n, minRun):
		end = min(start + minRun - 1, n - 1)
		insertionSort(arr, start, end)

	# Start merging from size RUN (or 32). It will merge
	# to form size 64, then 128, 256 and so on ....
	size = minRun
	while size < n:
		
		# Pick starting point of left sub array. We
		# are going to merge arr[left..left+size-1]
		# and arr[left+size, left+2*size-1]
		# After every merge, we increase left by 2*size
		for left in range(0, n, 2 * size):

			# Find ending point of left sub array
			# mid+1 is starting point of right sub array
			mid = min(n - 1, left + size - 1)
			right = min((left + 2 * size - 1), (n - 1))

			# Merge sub array arr[left.....mid] &
			# arr[mid+1....right]
			if mid < right:
				merge(arr, left, mid, right)

		size = 2 * size

arr = []


# The main function to sort
# an array of the given size
# using heapsort algorithm

def heapsort():
	global arr
	h = []

	# building the heap

	for value in arr:
		heappush(h, value)
	arr = []

	# extracting the sorted elements one by one

	arr = arr + [heappop(h) for i in range(len(h))]


# The main function to sort the data using
# insertion sort algorithm

def InsertionSort(begin, end):
	left = begin
	right = end

	# Traverse through 1 to len(arr)

	for i in range(left + 1, right + 1):
		key = arr[i]

		# Move elements of arr[0..i-1], that are
		# greater than key, to one position ahead
		# of their current position

		j = i - 1
		while j >= left and arr[j] > key:
			arr[j + 1] = arr[j]
			j = j - 1
		arr[j + 1] = key


# This function takes last element as pivot, places
# the pivot element at its correct position in sorted
# array, and places all smaller (smaller than pivot)
# to left of pivot and all greater elements to right
# of pivot

def Partition(low, high):
	global arr

# pivot

	pivot = arr[high]

# index of smaller element

	i = low - 1

	for j in range(low, high):

		# If the current element is smaller than or
		# equal to the pivot

		if arr[j] <= pivot:

			# increment index of smaller element

			i = i + 1
			(arr[i], arr[j]) = (arr[j], arr[i])
	(arr[i + 1], arr[high]) = (arr[high], arr[i + 1])
	return i + 1


# The function to find the median
# of the three elements in
# in the index a, b, d

def MedianOfThree(a, b, d):
	global arr
	A = arr[a]
	B = arr[b]
	C = arr[d]

	if A <= B and B <= C:
		return b
	if C <= B and B <= A:
		return b
	if B <= A and A <= C:
		return a
	if C <= A and A <= B:
		return a
	if B <= C and C <= A:
		return d
	if A <= C and C <= B:
		return d


# The main function that implements Introsort
# low --> Starting index,
# high --> Ending index
# depthLimit --> recursion level

def IntrosortUtil(begin, end, depthLimit):
	global arr
	size = end - begin
	if size < 16:

		# if the data set is small, call insertion sort

		InsertionSort(begin, end)
		return

	if depthLimit == 0:

		# if the recursion limit is occurred call heap sort

		heapsort()
		return

	pivot = MedianOfThree(begin, begin + size // 2, end)
	(arr[pivot], arr[end]) = (arr[end], arr[pivot])

	# partitionPoint is partitioning index,
	# arr[partitionPoint] is now at right place

	partitionPoint = Partition(begin, end)

	# Separately sort elements before partition and after partition

	IntrosortUtil(begin, partitionPoint - 1, depthLimit - 1)
	IntrosortUtil(partitionPoint + 1, end, depthLimit - 1)


# A utility function to begin the Introsort module

def Introsort(begin, end):

	# initialise the depthLimit as 2 * log(length(data))

	depthLimit = 2 * math.floor(math.log2(end - begin))
	IntrosortUtil(begin, end, depthLimit)

def countingSort(arr, exp1):

	n = len(arr)

	# The output array elements that will have sorted arr
	output = [0] * (n)

	# initialize count array as 0
	count = [0] * (10)

	# Store count of occurrences in count[]
	for i in range(0, n):
		index = arr[i] // exp1
		count[index % 10] += 1

	# Change count[i] so that count[i] now contains actual
	# position of this digit in output array
	for i in range(1, 10):
		count[i] += count[i - 1]

	# Build the output array
	i = n - 1
	while i >= 0:
		index = arr[i] // exp1
		output[count[index % 10] - 1] = arr[i]
		count[index % 10] -= 1
		i -= 1

	# Copying the output array to arr[],
	# so that arr now contains sorted numbers
	i = 0
	for i in range(0, len(arr)):
		arr[i] = output[i]

# Method to do Radix Sort
def radixSort(arr):

	# Find the maximum number to know number of digits
	max1 = max(arr)

	# Do counting sort for every digit. Note that instead
	# of passing digit number, exp is passed. exp is 10^i
	# where i is current digit number
	exp = 1
	while max1 / exp > 0:
		countingSort(arr, exp)
		exp *= 10


def main():
    elementsI = list()
    timesI = list()
    elementsQ = list()
    timesQ = list()
    elementsH = list()
    timesH = list()
    elementsT = list()
    timesT = list()
    global arr
    elementsIntro = list()
    timesIntro = list()
    elementsR = list()
    timesR = list()
    for i in range(1,10):

		# array acak
        a = randint(0,100*i,1000*i)

		# array menaik
        # b = randint(0,100*i,1000*i)
        # a = b
        # print('before => ', b)
        # insertion_sort(b)
        # print('after => ',b)

		# array menurun
        # b = randint(0,100*i,1000*i)
        # print('before => ', b)
        # insertion_sort(b)
        # print('after => ',b)
        # a = b[::-1]
        # print('after reverse => ',a)

        arr = a
        panjang = len(a)
        convert = list(map(int, a))

        # insertion sort
        startI = timeit.default_timer()
        insertion_sort(a)
        endI = timeit.default_timer()

        # quick sort
        startQ = timeit.default_timer()
        quick_sort(0,len(a)-1,a)
        endQ = timeit.default_timer()

        # heap sort
        startH = timeit.default_timer()
        HeapSort(a)
        endH = timeit.default_timer()

        # tim sort
        startT = timeit.default_timer()
        timSort(a)
        endT = timeit.default_timer()

        # intro sort
        startIntro = timeit.default_timer()
        Introsort(0, panjang-1)
        endIntro = timeit.default_timer()

        # radix sort
        startR = timeit.default_timer()
        radixSort(convert)
        endR = timeit.default_timer()

        print(len(a), "Elements Sorted by Insertionsort in ", endI-startI)
        print(len(a), "Elements Sorted by Quicksort in ", endQ-startQ)
        print(len(a), "Elements Sorted by HeapSort in ", endH-startH)
        print(len(a), "Elements Sorted by Timsort in ", endT-startT)
        print(len(a), "Elements Sorted by Introsort in ", endIntro-startIntro)
        print(len(a), "Elements Sorted by Radixsort in ", endR-startR)

        elementsI.append(len(a))
        timesI.append(endI-startI)
        elementsQ.append(len(a))
        timesQ.append(endQ-startQ)
        elementsH.append(len(a))
        timesH.append(endH-startH)
        elementsT.append(len(a))
        timesT.append(endT-startT)
        elementsIntro.append(len(a))
        timesIntro.append(endIntro-startIntro)
        elementsR.append(len(a))
        timesR.append(endR-startR)

    # plt.figure(figsize=(30, 20))
    plt.xlabel('List Length')
    plt.ylabel('Time Complexity')
    plt.plot(elementsI, timesI, label ='Insertion Sort',)
    plt.plot(elementsQ, timesQ, label ='Quick Sort',)
    plt.plot(elementsH, timesH, label ='Heap Sort',)
    plt.plot(elementsT, timesT, label ='Tim Sort',)
    plt.plot(elementsIntro, timesIntro, label ='Intro Sort',)
    plt.plot(elementsR, timesR, label ='Radix Sort',)
    plt.grid()
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
	main()