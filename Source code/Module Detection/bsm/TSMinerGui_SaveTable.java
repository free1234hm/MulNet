package bsm;

import javax.swing.*;

import java.awt.*;
import java.awt.event.*;
import java.io.*;

import javax.imageio.*;

import java.awt.image.*;

/**
 * Class to encapsulate window used to specify a file to save a TSMiner model
 */ 
public class TSMinerGui_SaveTable extends JPanel
{
	/**
	 * Class constructor
	 */
	public TSMinerGui_SaveTable(final JFrame theFrame, JTable table)
	{
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		String[] sznames = ImageIO.getWriterFormatNames();
		JFileChooser theChooser = new JFileChooser();
		add(theChooser);
		theChooser.setDialogType(JFileChooser.SAVE_DIALOG);
		//theChooser.addChoosableFileFilter(new JAVAFileFilter("jpg"));
		//theChooser.addChoosableFileFilter(new JAVAFileFilter("png"));
		theChooser.addActionListener(new ActionListener() 
		{
			public void actionPerformed(ActionEvent e) 
			{
				// set label's icon to the current image
				String state = (String)e.getActionCommand();

				if (state.equals(JFileChooser.CANCEL_SELECTION))
				{
					theFrame.setVisible(false);
					theFrame.dispose();
				}
				else if (state.equals(JFileChooser.APPROVE_SELECTION))
				{
					File file = theChooser.getSelectedFile();
					try
					{
						PrintWriter pw = new PrintWriter(new FileOutputStream(file));
						for(int i=0;i<table.getRowCount();i++){
							for(int j=0;j<table.getColumnCount();j++){
								pw.print(table.getValueAt(i, j)+"\t");
							}
							pw.println();
						}
						pw.close();
					    
					}catch (final IOException fex){
						javax.swing.SwingUtilities.invokeLater(new Runnable() 
						{
							public void run() 
							{
								JOptionPane.showMessageDialog(null, fex.getMessage(), 
										"Exception thrown", JOptionPane.ERROR_MESSAGE);
							}
						});
						fex.printStackTrace(System.out);
					}
					theFrame.setVisible(false);
					theFrame.dispose();
				}
			}
		});			      
	}
	/*
	class JAVAFileFilter extends FileFilter {
	    String ext;

	    public JAVAFileFilter(String ext) {
	        this.ext = ext;
	    }

	   
	    public boolean accept(File file) {
	        if (file.isDirectory()) {
	            return true;
	        }
	        String fileName = file.getName();
	        int index = fileName.lastIndexOf('.');
	        if (index > 0 && index < fileName.length() - 1) {
	            // ��ʾ�ļ����Ʋ�Ϊ".xxx"��"xxx."֮����
	            String extension = fileName.substring(index + 1).toLowerCase();
	            // ����ץ�����ļ���չ����������������Ҫ��ʾ����չ��(������extֵ),�򷵻�true,��ʾ�����ļ���ʾ����,���򷵻�
	            // true.
	            if (extension.equals(ext))
	                return true;
	        }
	        return false;
	    }

	    // ʵ��getDescription()����,���������ļ���˵���ַ���!!!
	    public String getDescription() {
	        if (ext.equals("png")) return "ͼƬ(*.png)";
	        if (ext.equals("jpg")) return "ͼƬ(*.jpg)";
	        return "";
	    }
	}
	*/
}