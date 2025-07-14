#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> arr;
    int num;
    
    cout << "请输入整数数组，用空格分隔，按回车结束输入：" << endl;
    
    // 持续读取输入直到遇到换行符
    while (cin >> num) {
        arr.push_back(num);
        // 检查下一个字符是否是换行符
        if (cin.get() == '\n') {
            break;
        }
    }
    
    // 输出结果
    cout << "输入的数组为：";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;
    
    return 0;
}