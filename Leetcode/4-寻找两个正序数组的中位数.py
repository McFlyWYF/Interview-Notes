'''
将list1和list2合并为一个list，再对list进行排序求中位数
'''

class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        combine = nums1 + nums2
        combine = sorted(combine)

        length = len(combine)
        if (length % 2 == 0):
            return float(combine[int((length / 2) - 1)] + combine[int(length / 2)]) / 2
        else:
            return combine[int(length / 2)]
