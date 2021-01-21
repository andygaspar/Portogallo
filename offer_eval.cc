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


        void print_mat(){ 
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


        bool check_couple_condition(short* flights){
            if (flights[0]==3 and flights[1]==12 and  flights[2]==2 and  flights[3]==5){
                for (short i = 0; i< couples_rows; i++){
                    std::cout<<"siamo qua"<<std::endl;
                    // first airline eta check
                    if (mat[flights[0]][1] <= mat[flights[couples[i][0]]][0]){
                        std::cout<<"primo if"<<std::endl;
                        if (mat[flights[1]][1] <= mat[flights[couples[i][1]]][0]){
                            std::cout<<"secondo if"<<std::endl;
                            // couples[i]hecouples[i]k first airline's couples[i]onveniencouples[i]e
                            if (mat[flights[0]][ 2 + flights[0]] + mat[flights[1]][ 2 + flights[1]] > 
                                    mat[flights[0]][ 2 + flights[couples[i][0]]] + mat[flights[1]][ 2 + flights[couples[i][1]]]){
                                                std::cout<<"terzo if"<<std::endl;
                                // secouples[i]ond airline eta couples[i]hecouples[i]k
                                if (mat[flights[2]][1] <= mat[flights[couples[i][2]]][0]){

                                    std::cout<<"quarto if"<<std::endl;
                                    if (mat[flights[3]][1] <= mat[flights[couples[i][3]]][0]){

                                        std::cout<<"quinto if"<<std::endl;
                                        std::cout<<mat[flights[2]][2 + flights[2]] + mat[flights[3]][2 + flights[3]]<<" "<<mat[flights[2]][2 + flights[couples[i][2]]] + mat[flights[3]][2 + flights[couples[i][3]]]<<std::endl;
                                        if (mat[flights[2]][2 + flights[2]] + mat[flights[3]][2 + flights[3]] > 
                                                mat[flights[2]][2 + flights[couples[i][2]]] + 
                                                mat[flights[3]][2 + flights[couples[i][3]]]){

                                                    std::cout<<"trovato"<<std::endl;
                                                    return true;
                                                }
                                            
                                    }
                                }
                            }
                        }
                    }
                }
            }
            return false;
        }


        int air_couple_check(short* airl_pair){
            int matches = 0;
            for (int k = 0; k < 315; k++){
                if (check_couple_condition(&airl_pair[k*4]))
                        matches++;
            }
            std::cout<<"trovati "<<matches<<std::endl;
            //print_mat();
            return matches;
            
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
    {  return new OfferChecker(schedule_mat,row, col, coup, coup_row, coup_col, trip, trip_row, trip_col); } 
    int air_couple_check_(OfferChecker* off,short* airl_pair) {return off->air_couple_check(airl_pair);}

    bool check_couple_condition_(OfferChecker* off,short* flights) {return off->check_couple_condition(flights);}
    
    void print_mat_(OfferChecker* off){ off -> print_mat(); }
    void print_couples_(OfferChecker* off){ off -> print_couples(); } 
    void print_triples_(OfferChecker* off){ off -> print_triples(); } 

}
