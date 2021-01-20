#include <iostream> 
#include <string>
#include <vector>

int value = 0;

class OfferChecker{ 
    long** mat;
    int rows;
    int cols;

    public: 

       

        OfferChecker(long* vect, int row, int col): 
            mat{new long*[row]}, rows{row}, cols{col}
         {
         for (int i = 0 ; i< row; i++) {
                mat[i]= &vect[i*col];
            }
         }

        ~OfferChecker(){mat = nullptr;}

        void assign_vect(long** vect, int row, int col){ 
            std::cout<<" ciccio";
            for (int i = 0 ; i< row; i++) {
                for (int j=0; j< col; j++)
                    {std::cout<<vect[i][j]<<" ";}
            }
        }

        void myFunction(int val){ 
            std::cout <<val<< " Hello Geek!!!  " <<value<< std::endl; 
            value = 5;
        } 

        void printValue(){ 
            std::cout << "valore "<<value << std::endl; 
        } 

        void print_vect(){ 
            for (int i = 0 ; i< rows; i++) {
                for (int j=0; j< cols; j++)
                    {std::cout<<mat[i][j]<<" ";}
                    std::cout<<std::endl;
            }
            
            

        } 

}; 
int main() 
{ 
     std::cout << "iniziato" << std::endl; 
    // Creating an object 
    //OfferChecker t = OfferChecker(NULL, 0,0);  
  
    // Calling function 
    //t.myFunction(val);   
    std::cout << "finito" << std::endl; 
    return 0; 
} 


extern "C" { 
    OfferChecker* OfferChecker_(long* vect, int row, int col){ return new OfferChecker(vect,row, col); } 
    void Geek_myFunction(OfferChecker* geek, int val){ geek -> myFunction(val); } 
    void Geek_myValue(OfferChecker* geek){ geek -> printValue(); } 
    void print_vect_(OfferChecker* geek){ geek -> print_vect(); } 
}
