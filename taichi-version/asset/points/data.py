with open('pdata.txt') as file:
        points = []
        k = 0
        c = 0
        while 1:
            c = c+1
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            points.append([float(strs[0]), float(strs[1]), float(strs[2])])
            if c == 441:
                c = 0
                k = k + 1
                with open(str(k)+'.txt', 'w') as f:
                    for i in range(len(points)):
                        print(f'{points[i][0]} {points[i][1]} {points[i][2]}', file=f)
                points = []
