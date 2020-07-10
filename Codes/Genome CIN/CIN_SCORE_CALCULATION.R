###load packages###
library(TCGAbiolinks)
library(data.table)
library(plyr)
library(dplyr)
library(stringr)

########CIN score calculation#########
#---Download Copy Number Variation (CNV) data from TCGA to calculate genome CIN scores
query_cin_seg<-GDCquery(project = "TCGA-BRCA",
                        data.category = "Copy Number Variation",
                        data.type = "Copy Number Segment")
GDCdownload(query_cin_seg)
data<-GDCprepare(query_cin_seg)

#---calculate fraction segment altered
#CIN score=altered length/total length
data<-data%>%mutate(length=End-Start,Segment_Alter=if_else(abs(Segment_Mean)<=0.2,0,1))
#total length
Total<-data%>%group_by(Sample)%>%summarise(Total_Length=sum(length))
#alter length
Alter<-data%>%filter(Segment_Alter==1)%>%group_by(Sample)%>%summarise(Alter_Length=sum(length))
#Get cin score in primary tumor sites with sample names of TCGA-XX-XXXX-01X
CIN_SCORE<-left_join(Total,Alter)%>%
  mutate(Cin_Score=Alter_Length/Total_Length,group=substring(Sample,14,15))%>%
  filter(group=="01")%>%
  mutate(Barcode=substring(Sample,1,12),group=substring(Sample,14,16))%>%
  group_by(Barcode)%>%mutate(n=n())
#some sample have two CIN scores, use 01A and pick the max CIN score
CIN_DUP<-CIN_SCORE%>%filter(n>1 & group=='01A')%>%
  dplyr::select(-n)%>%group_by(Barcode)%>%
  slice(which.max(Total_Length))
CIN_UNIQUE<-CIN_SCORE%>%filter(n==1)%>%select(-n)
CIN_SCORE<-rbind(CIN_UNIQUE,CIN_DUP)%>%select(Barcode,Cin_Score)


