# List comprehension yay
# Task: given 3 int x, y, z print all possible coordinates given by (i, j, k) 
#       where i + j + k is not equal to n

def main():
    x = int(input("First num: "))
    y = int(input("Second num: "))
    z = int(input("Third num: "))
    n = int(input("Compare: "))

    allCoordinates = []

    for i in range(x + 1): 
        for j in range(y + 1): 
            for k in range(z + 1): 
                allCoordinates.append([i,j,k])
                
    print(f"All permutations of the list {i, j, k} is: ", allCoordinates) 
    # Output: [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    
    print(f"The length of allCoordinates is:", len(allCoordinates))
    # Output: 8
    
    for i in range(len(allCoordinates)):
        print(i, end=' ')
    

    # Iterate through all the list in allCoordinates - done
    # Sum individually and append to a new list
    # Compare the sum with n
    # If sum = n then remove the list
    # Repeat until all elements are compared
if __name__ == '__main__':
    main()