#include <iostream> 
#include <string>
#include <vector>

int value = 0;

class OfferChecker{ 
    double** mat;
    int mat_rows;
    int mat_cols;

    short** couples;
    int couples_rows;
    int couples_cols;

    short** triples;
    int triples_rows;
    int triples_cols;

    public: 

       

        OfferChecker(double* schedule_mat, int row, int col, short* coup, int coup_row, int coup_col, short* trip, int trip_row, int trip_col): 
            mat{new double*[row]}, mat_rows{row}, mat_cols{col}, couples{new short*[coup_row]}, couples_rows{coup_row}, couples_cols{coup_col}, triples{new short*[trip_row]}, triples_rows{trip_row}, triples_cols{trip_col}
         {
         for (int i = 0 ; i< row; i++) {
                mat[i]= &schedule_mat[i*col];
            }

        for (int i = 0 ; i< coup_row; i++) {
                couples[i]= &coup[i*coup_col];
            }

        for (int i = 0 ; i< trip_row; i++) {
                triples[i]= &trip[i*trip_col];
            }
         }

        ~OfferChecker(){mat = nullptr;}


        void print_vect(){ 
            for (int i = 0 ; i< mat_rows; i++) {
                for (int j=0; j< mat_cols; j++)
                    {std::cout<<mat[i][j]<<" ";}
                    std::cout<<std::endl;
            }
        }

        void print_couples(){ 
            for (int i = 0 ; i< couples_rows; i++) {
                for (int j=0; j< couples_cols; j++)
                    {std::cout<<couples[i][j]<<" ";}
                    std::cout<<std::endl;
            }
        }

        void print_triples(){ 
            for (int i = 0 ; i< triples_rows; i++) {
                for (int j=0; j< triples_cols; j++)
                    {std::cout<<triples[i][j]<<" ";}
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
    OfferChecker* OfferChecker_(double* schedule_mat, int row, int col, short* coup, int coup_row, int coup_col, short* trip, int trip_row, int trip_col)
    { return new OfferChecker(schedule_mat,row, col, coup, coup_row, coup_col, trip, trip_row, trip_col); } 
    void print_vect_(OfferChecker* off){ off -> print_vect(); }
    void print_couples_(OfferChecker* off){ off -> print_couples(); } 
    void print_triples_(OfferChecker* off){ off -> print_triples(); } 
}
