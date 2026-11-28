def equality_matrix(A=None, B=None):

    if A is None or B is None:
        return False

    if len(A) != len(B):
        return False

    for i in range(len(A)):
        if len(A[i]) != len(B[i]):
            return False

        for j in range(len(A[i])):
            if A[i][j] != B[i][j]:
                return False

    return True


A = [[1, 2], [3, 4]]

B = [[1, 2], [3, 4]]

print(equality_matrix(A, B))
A = [[1, 2], [3, 4]]
B = [[1, 2], [3, 5]]

print(equality_matrix(A, B))