
import java.awt.BorderLayout;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.gui.treevisualizer.PlaceNode2;
import weka.gui.treevisualizer.TreeVisualizer;
public class DecisionTree {
	
	public static void main(String[] args) throws Exception {
	    Instances data = DataSource.read("/Users/ling/Documents/Eclipseworkspace/Weka/NewPattern/src/test/hasValueLogic.csv");
	    if (data.classIndex() == -1)
	    	data.setClassIndex(data.numAttributes() - 1);
	    List<String> options = new ArrayList<>();
		options.add("-U"); // pruned tree
//		options.add("-C 1");         // confidence threshold for pruning. (Default: 0.25)
//	    options.add("-M 1");            // minimum number of instances per leaf. (Default: 2)
//	    options.add("-R");            // use reduced error pruning. No subtree raising is performed.
//	    options.add("-N 1");            // number of folds for reduced error pruning. One fold is used as the pruning set. (Default: 3)
	    //options.add("-B");            // Use binary splits for nominal attributes.
	    //options.add("-S");            // not perform subtree raising.
	    //options.add("-L");            // not clean up after the tree has been built.
	    //options.add("-A");            // if set, Laplace smoothing is used for predicted probabilites.
	    //options.add("-Q");            // The seed for reduced-error pruning.
	    
	    System.out.println(data.classAttribute());
	    J48 tree = new J48();
	    //tree.setOptions(options.toArray(new String[options.size()]));		    
		tree.buildClassifier(data);
		System.out.print("train end");
			
		final Evaluation eval = new Evaluation(data);
		eval.evaluateModel(tree, data);
		    
		System.out.print("test end");
		System.out.println(eval.toSummaryString());
		System.out.println(eval.toMatrixString());
		System.out.println(tree.toString());
		visualize(tree);
		
	}
	public static void visualize(J48 tree) throws Exception{
		// display classifier
	     final javax.swing.JFrame jf = 
	       new javax.swing.JFrame("Weka Classifier Tree Visualizer: J48");
	     jf.setSize(800,400);
	     jf.getContentPane().setLayout(new BorderLayout());
	     TreeVisualizer tv = new TreeVisualizer(null,((J48)tree).graph(),new PlaceNode2());
	     jf.getContentPane().add(tv, BorderLayout.CENTER);
	     jf.addWindowListener(new java.awt.event.WindowAdapter() {
	       public void windowClosing(java.awt.event.WindowEvent e) {
	         jf.dispose();
	       }
	     });
	 
	     jf.setVisible(true);
	     tv.fitToScreen();
	}
}
