
def dataCorreleation(plt,sns,df):

                dataFramewithoutlabels = df.drop('Label', axis=1) 
                
                correlation_matrix = dataFramewithoutlabels.corr(numeric_only=True)
               
                
          
              
              

                # condition plot 
                baseline = df[df['Label'] == -1]
                threat = df[df['Label'] == 209]
                anger = df[df['Label'] == 208]
                highapproachPositiveEmotions = df[df['Label'] == 309]
                lowapproachPositiveEmotions = df[df['Label'] == 308]
                neutral= df[df['Label'] == 108]
                
                
                graph_name = "HRV_LF"
                plt.figure(figsize=(10, 6))
                plt.hist(baseline[graph_name], bins=30, alpha=0.5, label='baseline')
                plt.hist(threat[graph_name], bins=30, alpha=0.5, label='threat')
                plt.hist(anger[graph_name], bins=30, alpha=0.5, label='anger')
                plt.hist(highapproachPositiveEmotions[graph_name], bins=30, alpha=0.5, label='highapproachPositiveEmotions')
                plt.hist(lowapproachPositiveEmotions[graph_name], bins=30, alpha=0.5, label='lowapproachPositiveEmotions')
                plt.hist(neutral[graph_name], bins=30, alpha=0.5, label='neutral')


                plt.xlabel('feature values')
                plt.ylabel('frequency')
                plt.title('Histogram of ' + graph_name + ' by condition')
                plt.legend()
                plt.show()

            