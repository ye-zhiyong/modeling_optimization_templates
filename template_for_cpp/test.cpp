#include <iostream>
#include <vector>
using namespace std;

int main() {
    vector<int> arr;
    int num;
    
    cout << "�������������飬�ÿո�ָ������س��������룺" << endl;
    
    // ������ȡ����ֱ���������з�
    while (cin >> num) {
        arr.push_back(num);
        // �����һ���ַ��Ƿ��ǻ��з�
        if (cin.get() == '\n') {
            break;
        }
    }
    
    // ������
    cout << "���������Ϊ��";
    for (int x : arr) {
        cout << x << " ";
    }
    cout << endl;
    
    return 0;
}