# 经典的面试题
# 暴力法（超时）
# class Solution(object):
#     def maxArea(self, height):
#         """
#         :type height: List[int]
#         :rtype: int
#         """
        # def maxarea(i , j):
        #     if height[i] >= height[j]:
        #         return height[j] * (j - i)
        #     else:
        #         return height[i] * (j - i)
        #
        # max = 0
        # for i in range(len(height)):
        #     for j in range(i + 1, len(height)):
        #         mm = maxarea(i, j)
        #         if max < mm:
        #             max = mm
        #
        # return max

# 双指针法
'''
一个指针指向前，一个指向末尾，先找到最小的元素乘以x计算面积，再移动最小元素的指针，直到i <= j
'''
class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        i = 0
        j = len(height) - 1
        max_area = 0

        while(i <= j):
            m = min(height[i], height[j])
            area = m * (j - i)
            if max_area < area:
                max_area = area
            if height[i] < height[j]:
                i += 1
            else:
                j -= 1
        return max_area

l = [4, 3, 2, 1, 4]
s = Solution()
m = s.maxArea(l)
print(m)