# include "csv.h"
#include <iostream>
int main(){
  io::CSVReader<10> in("./out.csv");
  in.read_header(io::ignore_extra_column,"timestamp",
            "panda_joint1",
            "panda_joint2",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint1",
            "panda_joint3",
            "panda_finger_joint2");
  std::string timestamp; double panda_joint1; 
    double panda_joint2; double panda_joint4; 
    double panda_joint5; double panda_joint6;
    double panda_joint7; double panda_finger_joint1;
    double panda_joint3; double panda_finger_joint2;     
  while(in.read_row(timestamp,panda_joint1,panda_joint2,panda_joint4,panda_joint5,
  	panda_joint6,panda_joint6, panda_finger_joint1, panda_joint3, panda_finger_joint2)){
    // do stuff with the data
	std::cout<<timestamp;//panda_joint2;
	std::cout<<"\n";
  }
}