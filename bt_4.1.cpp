#include <iostream>
bool leap_year (int y);

int main(){
    int y = 0;
    std::cout << "Nhap nam: ";
    std::cin >> y;
    if(leap_year(y))
        std::cout << "La Nam nhuan/n";
    else
        std::cout << "Khong la Nam nhuan/n";

    return 0;
}

bool leap_year (int y){
    if(y % 4 != 0){
        return false;
    }
    if(y % 100 != 0){
        return true;
    }
    
    if (y % 400 != 0){
        return false;
    }
    return true;
}
