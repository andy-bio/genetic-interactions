#!/bin/bash
#### Este script recibe una carpeta donde estan todos los clusters en txt,
#### crea otra dentro con los resultados del enriquecimiento

cluster_folder=$(ls $1)

for i in $cluster_folder
    do
        for a in $(ls $1/${i})
            do
               find_enrichment.py $1/${i}/${a} ~/MyStuff/tesis/Jupyter/GO/population.txt ~/MyStuff/tesis/Jupyter/GO/pairs.txt --obo ~/MyStuff/tesis/Jupyter/GO/go.obo > $1/${i}/${a}_result.txt
            done
        
        mkdir $1/${i}/results
        mv $1/${i}/*result.txt $1/${i}/results

        mkdir $1/${i}/results_parsed

        for b in $(ls $1/${i}/results)
            do
                tail -n +90 $1/${i}/results/${b} | head -n 15 > $1/${i}/results_parsed/${b}_parsed.txt
            done
            
    done



