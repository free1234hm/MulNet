package heatmapframe;

import javax.swing.*;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Arrays;

/**
 * <p>This class is a very simple example of how to use the HeatMap class.</p>
 *
 * <hr />
 * <p><strong>Copyright:</strong> Copyright (c) 2007, 2008</p>
 *
 * <p>HeatMap is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.</p>
 *
 * <p>HeatMap is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.</p>
 *
 * <p>You should have received a copy of the GNU General Public License
 * along with HeatMap; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA</p>
 *
 * @author Matthew Beckler (matthew@mbeckler.org)
 * @author Josh Hayes-Sheen (grey@grevian.org), Converted to use BufferedImage.
 * @author J. Keller (jpaulkeller@gmail.com), Added transparency (alpha) support, data ordering bug fix.
 * @version 1.6
 */

class HeatMapFrame extends JFrame
{
    HeatMap panel;

    public HeatMapFrame(double[][] data) throws Exception
    {
        super("BSM HeatMapFrame");
        Gradient gra = new Gradient();
        data = transpose(data);
        boolean useGraphicsYAxis = true;
        
        // you can use a pre-defined gradient:
        panel = new HeatMap(data, useGraphicsYAxis, Gradient.GRADIENT_BLUE_TO_RED);
        
        // or you can also make a custom gradient:
        Color[] gradientColors = new Color[]{Color.decode("#00BFFF"), Color.white, Color.red};
        //Color[] gradientColors = new Color[]{Color.white, Color.black};
        Color[] customGradient = Gradient.createMultiGradient(gradientColors, 500);
        panel.updateGradient(customGradient);
        
        // set miscelaneous settings
        panel.setDrawLegend(true);

        panel.setTitle("Height (m)");
        panel.setDrawTitle(false);

        panel.setXAxisTitle("Time Points");
        panel.setDrawXAxisTitle(false);

        panel.setYAxisTitle("Gene Index");
        panel.setDrawYAxisTitle(false);

        panel.setCoordinateBounds(0, data.length, 0, data[0].length);
        panel.setDrawXTicks(true);
        panel.setDrawYTicks(true);

        panel.setColorForeground(Color.black);
        panel.setColorBackground(Color.white);

        this.getContentPane().add(panel);
    }
    
    public HeatMap getHeatMap(){  
        return panel;       
    }
    
    // this function will be run from the EDT
    private static void createAndShowGUI() throws Exception
    {
    	int row=0;
        int column=0;
        String[][] value=null;
		try {
			BufferedReader bufferedReader1 = new BufferedReader(new FileReader("MDRM_AA_20true3child.txt"));
			
                String lineTxt = null;
                
                while((lineTxt = bufferedReader1.readLine()) != null){                                       	
                   if(column<Arrays.asList(lineTxt.split("\\t")).size()) column=Arrays.asList(lineTxt.split("\\t+")).size();
                   row++;
                }
                bufferedReader1.close();
                
                String value1[][] = new String[row][column];
                String genes[] = new String[row];
                
                BufferedReader bufferedReader2 = new BufferedReader(new FileReader("MDRM_AA_20true3child.txt"));
                int count=0;
                while((lineTxt = bufferedReader2.readLine()) != null){  
                	value1[count][0] = (count+1)+"";
                    		for(int i=1;i<Arrays.asList(lineTxt.split("\\t")).size();i++){
                    			value1[count][i]=Arrays.asList(lineTxt.split("\\t")).get(i);
                    		}   
                    		genes[count] = Arrays.asList(lineTxt.split("\\t")).get(0);
                    		count++;
                }
                bufferedReader2.close();
                value=value1;
                
    } catch (Exception e) {
        System.out.println("��ȡ�ļ����ݳ���");
        e.printStackTrace();
    }
    	double[][] data = new double[value.length][value[0].length-2];
    	for(int i=0;i<data.length;i++){
    		for(int j=0;j<data[0].length;j++){
    			if(value[i][j+2] != null && value[i][j+2].length()>0){
    				data[i][j] = Double.parseDouble(value[i][j+2]);
    			}			
    		}
    	}
        HeatMapFrame hmf = new HeatMapFrame(data);
        hmf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        hmf.setSize(500,500);
        hmf.setVisible(true);
    }
    
    public double[][] transpose(double[][] a){
		double b[][] = new double[a[0].length][a.length];
		for(int i=1;i<=b.length;i++){
			for(int j=1;j<=b[0].length;j++){
				b[i-1][j-1] = a[j-1][i-1];
			}
		}
		return b;
	}
    public static void main(String[] args)
    {
        SwingUtilities.invokeLater(new Runnable()
        {
            public void run()
            {
                try
                {
                    createAndShowGUI();
                }
                catch (Exception e)
                {
                    System.err.println(e);
                    e.printStackTrace();
                }
            }
        });
    }
}

