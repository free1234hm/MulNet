package bsm;

import java.util.List;
import bsm.core.StatUtil;

public class EnrichmentAnalysis {

	public String[][] enrichment(List<Integer> posilist, List<Integer> negalist,
			String[] regNames, int[][] bindingdata, int numgenes){
		String[][] table = new String[regNames.length+1][8];
		table[0][0] = "Name";
		table[0][1] = "Gene Number";
		table[0][2] = "Posi_Overlaps";
		table[0][3] = "Posi_Pvalue";
		table[0][4] = "Nega_Overlaps";
		table[0][5] = "Nega_Pvalue";
		table[0][6] = "Overlaps";
		table[0][7] = "Pvalue";
		for(int i=0;i<regNames.length;i++){
			int[] tfgene = bindingdata[i];
			if(tfgene != null && tfgene.length > 0){
				int posi_overlap = 0;
				int nega_overlap = 0;
				int all_overlap = 0;
				for(int tg:tfgene){
					if(posilist.contains(tg)){
						posi_overlap++;
						all_overlap++;
					}else if(negalist.contains(tg)){
						nega_overlap++;
						all_overlap++;
					}
				}
				double posi_pvalue = StatUtil.hypergeometrictail(posi_overlap-1, tfgene.length, 
						numgenes-tfgene.length, posilist.size());
				double nega_pvalue = StatUtil.hypergeometrictail(nega_overlap-1, tfgene.length, 
						numgenes-tfgene.length, negalist.size());  
				double pvalue = StatUtil.hypergeometrictail(all_overlap-1, tfgene.length, 
						numgenes-tfgene.length, (posilist.size()+negalist.size()));
				table[i+1][0] = regNames[i];
				table[i+1][1] = tfgene.length+"";
				table[i+1][2] = posi_overlap+"";
				table[i+1][3] = posi_pvalue+"";
				table[i+1][4] = nega_overlap+"";
				table[i+1][5] = nega_pvalue+"";
				table[i+1][6] = all_overlap+"";
				table[i+1][7] = pvalue+"";
			}else{
				table[i+1][0] = regNames[i];
				table[i+1][1] = 0+"";
				table[i+1][2] = 0+"";
				table[i+1][3] = 1+"";
				table[i+1][4] = 0+"";
				table[i+1][5] = 1+"";
				table[i+1][6] = 0+"";
				table[i+1][7] = 1+"";
			}
		}
		return table;
	}
	
}
