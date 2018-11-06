package org.simbrain.custom_sims.simulations.sorn_abridging;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.*;
import java.io.File;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;

import javax.swing.JTabbedPane;
import javax.swing.JTextField;
import javax.swing.JButton;
import javax.swing.JToggleButton;
import java.awt.geom.Point2D;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.Color;

import org.simbrain.custom_sims.helper_classes.PlotBuilder;


import org.simbrain.custom_sims.RegisteredSimulation;
import org.simbrain.custom_sims.helper_classes.ControlPanel;
import org.simbrain.custom_sims.helper_classes.NetBuilder;
import org.simbrain.custom_sims.helper_classes.OdorWorldBuilder;
import org.simbrain.custom_sims.helper_classes.PlotBuilder;
import org.simbrain.workspace.CouplingManager;
import org.simbrain.network.gui.NetworkPanel;
import org.simbrain.util.projection.Projector;
import org.simbrain.plot.projection.ProjectionModel;
import org.simbrain.util.projection.DataColoringManager;
import org.simbrain.plot.projection.ProjectionComponent;
import org.simbrain.plot.projection.DataPointColoringDialog;
// import org.simbrain.custom_sims.helper_classes.AddSTDPRule;
// import org.simbrain.custom_sims.helper_classes.SORNNeuronRule;

import org.simbrain.network.NetworkComponent;
import org.simbrain.network.core.Network;
import org.simbrain.network.listeners.NetworkEvent;
import org.simbrain.network.core.Neuron;
import org.simbrain.network.gui.nodes.NeuronNode;
import org.simbrain.network.listeners.NeuronListener;
import org.simbrain.network.connections.*;
import org.simbrain.network.core.*;
import org.simbrain.network.core.NeuronUpdateRule.InputType;
import org.simbrain.network.neuron_update_rules.interfaces.BiasedUpdateRule;
import org.simbrain.network.groups.*;
import org.simbrain.network.layouts.*;
import org.simbrain.network.desktop.*;
import org.simbrain.network.neuron_update_rules.*;
import org.simbrain.network.synapse_update_rules.*;
import org.simbrain.network.synapse_update_rules.spikeresponders.*;
import org.simbrain.network.update_actions.*;
import org.simbrain.util.randomizer.*;
import org.simbrain.util.*;
import org.simbrain.workspace.*;
import org.simbrain.util.SimbrainConstants.Polarity;
import javax.swing.JInternalFrame;
import javax.swing.JComboBox;
import org.simbrain.util.math.*;
import org.simbrain.workspace.gui.SimbrainDesktop;
import org.simbrain.world.odorworld.*;
import org.simbrain.world.odorworld.entities.*;
import org.simbrain.util.environment.*;
import org.simbrain.world.odorworld.entities.RotatingEntity;
import org.simbrain.world.odorworld.entities.OdorWorldEntity;
import org.simbrain.world.odorworld.sensors.TileSensor;
import org.simbrain.workspace.Coupling;
import org.simbrain.workspace.MismatchedAttributesException;
import org.simbrain.workspace.PotentialConsumer;
import org.simbrain.workspace.PotentialProducer;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import org.simbrain.world.odorworld.sensors.SmellSensor;

public class SORNAbridging extends RegisteredSimulation {

  public enum Stimulus {ABR, SCR}

  /** Network variables **/
  boolean homeoPlastic = true;
  boolean frozenWeights = false;
  int NUM_NEURONS = 150; // originally 2500
  int NUM_INPUTS = 6; // originally 200
  int GRID_SPACE = 25;
  int RADIUS = 400; // 100
  int EE_KIN = 10; // originally 25 (this is lambda_w?)
  int IE_KIN = (int)(0.2 * NUM_NEURONS/10);
  // int EI_KIN = NUM_NEURONS/50;
  int EI_KIN = 50;
  double TE_max = 0.75; //  0.5;
  double TI_max = 1.4; // 0.8
  double n_STDP = 0.001; // STDP learning Rate
  double t_factor = 0.01; // multiplier for relative time step
  double time_step = 1.0; // 41.38; // each update in network advances time by 'time_step' (ms)
  double etaIP = 0.001;
  int y_loc, x_loc;

  /** Stimulus variables **/
  Stimulus currentStim = Stimulus.ABR;
  Stimulus prevStim = Stimulus.ABR;
  OdorWorld motionWorld;
  OdorWorldComponent motionWorldComponent;
  WorkspaceComponent motionWorldWorkspace;
  NeuronGroup receptorNeurons, inputNeurons;
  CouplingManager couple;
  Workspace workspace;

  List<PotentialProducer> receptorProducers = new ArrayList<PotentialProducer>();
  List<PotentialConsumer> sornInputConsumers = new ArrayList<PotentialConsumer>();
  OdorWorldEntity abridgingEntity;
  // String bouncerDispersion = "20.00";
  // double bouncerDispDouble = 20.00;

  /** Abridging params / objects **/
  boolean occluded = true;
  boolean abridged = false;
  int motionSpeed = 10;
  int abrOccWidth = 202;
  int abrReceptorWidth = 101;
  int iterPerTarget = (int) abrReceptorWidth / motionSpeed;
  int abrHeading = 0;
  RotatingEntity mouseAbridging;
  double deflectFactor =  0.2 + 0.1 * Math.floor(motionSpeed / 15);

  /** Scrambling params / objects **/
  int currentTarget = 1;
  boolean scrambled = false;
  int scrmTargetWidth = 101;
  RotatingEntity mouseScrambling;
  int numTargets = 6;
  int[] scrmblPosition = new int[2];
  int scrmblHeading = 0;
  double scrmblSOA = 120.0;


  /** General motion world params / objects **/
  boolean workspaceRunning = false;
  boolean stimInterrupt = false;
  int worldHeight, worldWidth;
  int tileHeight = 80;
  NumberFormat doubleFormatter = new DecimalFormat("#0.00");
  // TileSensor occluderTile, scrambleTarg1, abridgeTargA;
  TileSensor targetA, targetB, targetO1, targetO2, targetD, targetE;
  TileSensor target1, target2, target3, target4, target5, target6;
  // Neuron scrm1, abrA, occluder;
  Neuron rec1, rec2, rec3, rec4, rec5, rec6;
  Coupling occluderCoupling, targACoupling, targ1Coupling;
  Coupling aCoupling, bCoupling, o1Coupling, o2Coupling, dCoupling, eCoupling;
  Coupling t1Coupling, t2Coupling, t3Coupling, t4Coupling, t5Coupling, t6Coupling;


  /** GUI variables **/
  boolean SORNvisible = true;
  ControlPanel controlPanel;
  JInternalFrame internalFrame = new JInternalFrame("Control panel", true, true);
  LabelledItemPanel panel = new LabelledItemPanel();
  JToggleButton freezeNet = new JToggleButton("Freeze", false);
  JButton unfreezeNet = new JButton("Unfreeze");
  JToggleButton homeostaticPlasticity = new JToggleButton("On/off", true);
  JButton abridging = new JButton("Toggle");
  JToggleButton occluderToggle = new JToggleButton("Occluded (C)", true);
  JTextField abrSpeedField = new JTextField("1.0");
  JTextField netTimeStep = new JTextField("1.0");
  JTextField scrSOAField = new JTextField("120.0"); // study used 75, 120 and 190 ms SOA
  String[] stimulusType = {"abridging", "scrambling"};
  JComboBox<String> stimulusSelect = new JComboBox<String>(stimulusType);

  /** Neural net variables **/
  Network SORNNetwork;
  Network stimulusNetwork;
  NetworkComponent SORNComponent;
  NetworkComponent stimulusComponent;
  ArrayList<Neuron> neurons = new ArrayList<Neuron>(); // excitatory neurons
  ArrayList<Neuron> inhibitoryNeurons = new ArrayList<Neuron>(); // inhibitory neurons
  ArrayList<Neuron> inNeurons = new ArrayList<Neuron>(); // input neurons
  SORNNeuronRule sornRule = new SORNNeuronRule(); // excitatory neuron rule
  SORNNeuronRule str = new SORNNeuronRule(); // inhibitory neuron rule
  AddSTDPRule stdp = new AddSTDPRule();
  NeuronGroup ng, ngIn, input; // excitatory/inhibitory/input neuron groups
  PolarizedRandomizer exRand, inRand;
  RadialSimpleConstrainedKIn ee_con, ie_con, ei_con;
  SynapseGroup sg_ee, sg_ie, sg_ei, input_ee, ee_input, input_ie, ie_input;
  Sparse input_ee_con, ee_input_con, input_ie_con, ie_input_con;

  /** Plot components **/
  PlotBuilder plotPCAEx, plotPCAIn;
  Projector plotPCAExProj, plotPCAInProj;
  DataColoringManager plotPCAExClrMgr, plotPCAInClrMgr;
  ProjectionComponent exPCAComponent, inPCAComponent;

  /**
  * Construct the simulation.
  *
  * @param desktop
  */
  public SORNAbridging(SimbrainDesktop desktop) {
    super(desktop);
  }

  public SORNAbridging() {
    super();
  }

  /**
  * Run the simulation
  */
  @Override
  public void run() {
    String FS = System.getProperty("file.separator");
    workspace = sim.getWorkspace();
    workspace.openWorkspace(new File("scripts" + FS + "scriptmenu" + FS + "SORNabridgingscramblingNEW.zip"));
    couple = workspace.getCouplingManager();
    setUpSORNNetwork();
    setUpSORNNeurons();
    setUpSORNSynapses();
    SORNNetworkInitialisation();
    setupStimulusNetwork();
    setupCoupling();
    setupListeners();
    makeControlPanel();
    setupGraphs();
  }


  @Override
  public String getName() {
    return "SORN-Abridging";
  }

  @Override
  public SORNAbridging instantiate(SimbrainDesktop desktop) {
    return new SORNAbridging(desktop);
  }

  void setUpSORNNetwork() {
    SORNComponent = new NetworkComponent("SORN Network");
    workspace.addWorkspaceComponent(SORNComponent);
    sim.getDesktop().getDesktopComponent(workspace.getComponent("SORN Network")).getParentFrame().setBounds(900,0,850,404);
    SORNNetwork = SORNComponent.getNetwork();
    SORNNetwork.setTimeStep(t_factor*time_step);
  }

  void fixPlotColoring() {
    plotPCAExClrMgr.setHotColor(Color.BLUE);
    plotPCAExClrMgr.setBaseColor(Color.RED);
    plotPCAInClrMgr.setHotColor(Color.RED);
    plotPCAInClrMgr.setBaseColor(Color.BLUE);
    plotPCAExClrMgr.setColoringMethod("DecayTrail");
    plotPCAInClrMgr.setColoringMethod("DecayTrail");
    plotPCAExClrMgr.setFloor(0.2);
    plotPCAInClrMgr.setFloor(0.2);
    plotPCAExClrMgr.setIncrementAmount(0.002);
    plotPCAInClrMgr.setIncrementAmount(0.002);
    // exPCAComponent.setGuiOn(true);
    // inPCAComponent.setGuiOn(true);
    exPCAComponent.update();
    inPCAComponent.update();
    // plotPCAExProj.iterate();
    // plotPCAInProj.iterate();
  }

  private void resetScrmMouse() {
    scrmblHeading = 0;
    currentTarget = 1;
    mouseScrambling.setCenterLocation((int) scrmTargetWidth/2, (int) (worldHeight - tileHeight/2));
    mouseScrambling.setHeading(0);
  }

  private void resetAbrMouse() {
    abrHeading = 0;
    mouseAbridging.setCenterLocation((int) abrReceptorWidth/2, (int) tileHeight/2);
    mouseAbridging.setHeading(0);
    mouseAbridging.setVelocityX(0);
  }

  private void setUpSORNNeurons() {
    /* EXCITATORY */
    sornRule.sethIP((2*NUM_INPUTS)/NUM_NEURONS);
    sornRule.setRefractoryPeriod(1);
    sornRule.setAddNoise(true);
    for (int i = 0; i < NUM_NEURONS; i++) {
      Neuron n = new Neuron(SORNNetwork);
      sornRule.setMaxThreshold(TE_max);
      sornRule.setThreshold(TE_max * Math.random() + 0.01);
      n.setPolarity(Polarity.EXCITATORY);
      n.setUpdateRule(sornRule.deepCopy());
      neurons.add(n);
    }

    /* INHIBITORY */
    for (int i = 0; i < (int) (NUM_NEURONS * 0.2); i++) {
      Neuron n = new Neuron(SORNNetwork);
      str.setThreshold(TI_max * Math.random() + 0.01);
      str.setEtaIP(etaIP);
      str.setRefractoryPeriod(1);
      str.setAddNoise(true);
      n.setPolarity(Polarity.INHIBITORY);
      n.setUpdateRule(str.deepCopy());
      inhibitoryNeurons.add(n);
    }

    ng = new NeuronGroup(SORNNetwork, neurons);
    ngIn = new NeuronGroup(SORNNetwork, inhibitoryNeurons);
    GridLayout layout = new GridLayout(GRID_SPACE, GRID_SPACE, (int) Math.sqrt(NUM_NEURONS));
    ng.setLabel("Excitatory population");
    SORNNetwork.addGroup(ng);
    ng.setLayout(layout);
    ng.applyLayout(new Point(10,10));

    layout = new GridLayout(GRID_SPACE, GRID_SPACE, (int) Math.sqrt(0.2 * NUM_NEURONS));
    ngIn.setLabel("Inhibitory population");
    SORNNetwork.addGroup(ngIn);
    ngIn.setLayout(layout);
    x_loc = (int) (Math.sqrt(NUM_NEURONS) * GRID_SPACE + 300);
    ngIn.applyLayout(new Point(x_loc, 10));

    exRand = new PolarizedRandomizer(Polarity.EXCITATORY, ProbDistribution.UNIFORM);
    inRand = new PolarizedRandomizer(Polarity.INHIBITORY, ProbDistribution.UNIFORM);
    exRand.setParam1(0);
    exRand.setParam2(1);
    inRand.setParam1(0);
    inRand.setParam2(1);

    for (Neuron n : neurons) {
      ((SORNNeuronRule) n.getUpdateRule()).init(n);
    }

    for (int i = 0; i < NUM_INPUTS; i++) {
      Neuron n = new Neuron(SORNNetwork);
      // SpikingThresholdRule inRule = new SpikingThresholdRule();
      // inRule.setThreshold(0.96);
      sornRule.setThreshold(TE_max * Math.random() + 0.01);
      n.setPolarity(Polarity.EXCITATORY);
      n.setUpdateRule(sornRule.deepCopy());
      inNeurons.add(n);
    }

    input = new NeuronGroup(SORNNetwork, inNeurons);
    LineLayout line_layout = new LineLayout();
    input.setLabel("Input");
    SORNNetwork.addGroup(input);
    input.setLayout(line_layout);

    y_loc = (int) (Math.sqrt(NUM_INPUTS) * GRID_SPACE + 200);
    input.applyLayout(new Point(x_loc, y_loc));


  }

  private void setUpSORNSynapses() {
    ee_con = new RadialSimpleConstrainedKIn(EE_KIN, RADIUS);
    ie_con = new RadialSimpleConstrainedKIn(IE_KIN, 100000);
    ei_con = new RadialSimpleConstrainedKIn(EI_KIN, 100000);
    input_ee_con = new Sparse(0.5, false, false); // first argument (density) was 0.05 in original SORN.bsh
    System.out.println("Created In-EE");
    ee_input_con = new Sparse(0.2, false, false); // density was originally 0.01
    System.out.println("Created EE-In");
    input_ie_con = new Sparse(0.2, true, false); // density was originally 0.01
    System.out.println("Created In-IE");
    ie_input_con = new Sparse(0.2, true, false); // density was originally 0.01 (needed to make this connection and I think input->ie denser, in particular, because there's so few of them. Modified the other two as well to maintain some degree of common proportionality with original densities)
    System.out.println("Created IE-In");

    sg_ee = SynapseGroup.createSynapseGroup(ng, ng, ee_con, 1.0, exRand, inRand); // (source, target, connectionManager, excitatoryRatio, excitatory randomiser, inhibitory randomiser)
    sg_ee.setLabel("EE Synapses");
    sg_ee.setSpikeResponder(new Step(), Polarity.BOTH);
    sg_ie = SynapseGroup.createSynapseGroup(ngIn, ng, ie_con, 1.0, exRand, inRand);
    sg_ie.setLabel("IE Synapses");
    sg_ie.setSpikeResponder(new Step(), Polarity.BOTH);
    sg_ei = SynapseGroup.createSynapseGroup(ng, ngIn, ei_con, 1.0, exRand, inRand);
    sg_ei.setLabel("EI Synapses");
    sg_ei.setSpikeResponder(new Step(), Polarity.BOTH);
    SORNNetwork.addGroup(sg_ee);
    SORNNetwork.addGroup(sg_ie);
    SORNNetwork.addGroup(sg_ei);

    stdp.setLearningRate(n_STDP);
    sg_ee.setLearningRule(stdp, Polarity.BOTH);

    input_ee = SynapseGroup.createSynapseGroup(input, ng, input_ee_con, 1.0, exRand, inRand);
    ee_input = SynapseGroup.createSynapseGroup(ng, input, ee_input_con, 1.0, exRand, inRand);
    input_ie = SynapseGroup.createSynapseGroup(input, ngIn, input_ie_con, 1.0, exRand, inRand);
    ie_input = SynapseGroup.createSynapseGroup(ngIn, input, input_ie_con, 1.0, exRand, inRand);
    input_ee.setLabel("Input -> Excitatory");
    input_ee.setLearningRule(stdp, Polarity.BOTH);
    input_ee.setSpikeResponder(new Step(), Polarity.BOTH);
    ee_input.setLabel("Excitatory -> Input");
    ee_input.setLearningRule(stdp, Polarity.BOTH);
    ee_input.setSpikeResponder(new Step(), Polarity.BOTH);
    input_ie.setLabel("Input -> Inhibitory");
    input_ie.setSpikeResponder(new Step(), Polarity.BOTH);
    ie_input.setLabel("Inhibitory -> Input");
    ie_input.setSpikeResponder(new Step(), Polarity.BOTH);
    SORNNetwork.addGroup(input_ee);
    SORNNetwork.addGroup(ee_input);
    SORNNetwork.addGroup(input_ie);
    SORNNetwork.addGroup(ie_input);

  }

  private void SORNNetworkInitialisation() {
    for (Neuron n : neurons) {
      n.normalizeInhibitoryFanIn();
    }

    for (Neuron n : neurons) {
      n.normalizeExcitatoryFanIn();
    }

    for (Neuron n : input.getNeuronList()) {
      n.normalizeInhibitoryFanIn();
      n.normalizeExcitatoryFanIn();
    }
  }

  void setupStimulusNetwork() {
    motionWorldWorkspace = workspace.getComponent("abridging and scrambling motion.xml");
    motionWorldComponent = (OdorWorldComponent) motionWorldWorkspace;
    motionWorld = motionWorldComponent.getWorld();
    worldHeight = motionWorld.getHeight();
    worldWidth = motionWorld.getWidth();
    scrmblPosition = new int[]{(int) scrmTargetWidth/2, (int) (worldHeight - tileHeight/2)};
    mouseAbridging = (RotatingEntity) motionWorld.getEntity("Entity_17");
    resetAbrMouse();
    mouseScrambling = (RotatingEntity) motionWorld.getEntity("Entity_16");
    resetScrmMouse();
    stimulusComponent = (NetworkComponent) workspace.getComponent("abridging and scrambling receptors.xml");
    stimulusNetwork = stimulusComponent.getNetwork();
    stimulusNetwork.setTimeStep(t_factor*time_step);
    receptorNeurons = (NeuronGroup) stimulusNetwork.getGroupByLabel("Receptors");
    rec1 = stimulusNetwork.getNeuronByLabel("1/A");
    rec2 = stimulusNetwork.getNeuronByLabel("2/B");
    rec3 = stimulusNetwork.getNeuronByLabel("3/O");
    rec4 = stimulusNetwork.getNeuronByLabel("4/O");
    rec5 = stimulusNetwork.getNeuronByLabel("5/D");
    rec6 = stimulusNetwork.getNeuronByLabel("6/E");

    // scrm1 = stimulusNetwork.getNeuronByLabel("1");
    // abrA = stimulusNetwork.getNeuronByLabel("A");

    // inputNeurons = SORNNetwork.getGroupByLabel("Input");
    // abridgingReceptorProducers = NetworkComponent.getNeuronGroupProducer(stimulusComponent, receptorNeurons, "getExternalActivations");
    // scramblingReceptorProducers = NetworkComponent.getNeuronGroupProducer(stimulusComponent, scramblingNeurons, "getExternalActivations");
    // abridgingReceptorConsumers = NetworkComponent.getNeuronGroupConsumer(stimulusComponent, abridgingNeurons, "setInputValues");
    // scramblingReceptorConsumers = NetworkComponent.getNeuronGroupConsumer(stimulusCompoment, scramblingNeurons, "setInputValues");

    // inputConsumers = NetworkComponent.getNeuronGroupConsumer(SORNComponent, input, "setInputValues");

  }

  void setupCoupling() {
    // occluderTile = (TileSensor) motionWorld.getSensor("Entity_17", "Sensor_6");
    // scrambleTarg1 = (TileSensor) motionWorld.getSensor("Entity_16", "Sensor_4");
    // abridgeTargA = (TileSensor) motionWorld.getSensor("Entity_17", "Sensor_4");

    receptorProducers.add(NetworkComponent.getNeuronGroupProducer(stimulusComponent, receptorNeurons, "getExternalActivations"));
    sornInputConsumers.add(NetworkComponent.getNeuronGroupConsumer(SORNComponent, input, "setInputValues"));
    try {
      workspace.coupleOneToOne(receptorProducers, sornInputConsumers);
    } catch (MismatchedAttributesException e) {
      System.out.println("MismatchedAttributesException");
    }

    targetA = (TileSensor) motionWorld.getSensor("Entity_17", "Sensor_4");
    targetB = (TileSensor) motionWorld.getSensor("Entity_17", "Sensor_5");
    targetO1 = (TileSensor) motionWorld.getSensor("Entity_17", "Sensor_6");
    targetO2 = (TileSensor) motionWorld.getSensor("Entity_17", "Sensor_10");
    targetD = (TileSensor) motionWorld.getSensor("Entity_17", "Sensor_8");
    targetE = (TileSensor) motionWorld.getSensor("Entity_17", "Sensor_9");

    target1 = (TileSensor) motionWorld.getSensor("Entity_16", "Sensor_4");
    target2 = (TileSensor) motionWorld.getSensor("Entity_16", "Sensor_5");
    target3 = (TileSensor) motionWorld.getSensor("Entity_16", "Sensor_6");
    target4 = (TileSensor) motionWorld.getSensor("Entity_16", "Sensor_7");
    target5 = (TileSensor) motionWorld.getSensor("Entity_16", "Sensor_8");
    target6 = (TileSensor) motionWorld.getSensor("Entity_16", "Sensor_9");

    aCoupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(targetA, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec1, "setInputValue", double.class));
    bCoupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(targetB, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec2, "setInputValue", double.class));
    o1Coupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(targetO1, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec3, "setInputValue", double.class));
    o2Coupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(targetO2, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec4, "setInputValue", double.class));
    dCoupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(targetD, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec5, "setInputValue", double.class));
    eCoupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(targetE, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec6, "setInputValue", double.class));

    t1Coupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(target1, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec1, "setInputValue", double.class));
    t2Coupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(target2, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec2, "setInputValue", double.class));
    t3Coupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(target3, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec3, "setInputValue", double.class));
    t4Coupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(target4, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec4, "setInputValue", double.class));
    t5Coupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(target5, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec5, "setInputValue", double.class));
    t6Coupling = new Coupling(motionWorldComponent.getAttributeManager().createPotentialProducer(target6, "getValue", double.class), stimulusComponent.getAttributeManager().createPotentialConsumer(rec6, "setInputValue", double.class));

  }


  void runScramblingTrial() {

    Executors.newSingleThreadExecutor().execute(new Runnable() {
      @Override
      public void run() {
        scrmblSOA = Double.parseDouble(scrSOAField.getText());
        stimInterrupt = false;
        resetScrmMouse();
        mouseAbridging.setVelocityX(0);
        // workspace.stop();
        int count = 0;
        //int iterPerStep = (int) (scrmblSOA/time_step); // calculating SOA in terms of iterations
        int iterPerStep = (int) (scrmblSOA*0.05);

        while (!stimInterrupt) {
          // while (!Thread.currentThread().isInterrupted()) {
          count++;
          if (count == iterPerStep) {
            count = 0;
            switch (currentTarget) {
              case 1:
              currentTarget = 2;
              break;
              case 2:
              if (scrmblHeading == 180) {
                currentTarget = 1;
                scrmblHeading = 0;
              } else {
                currentTarget = scrambled ? 4 : 3;
              }
              break;
              case 3:
              if (scrmblHeading == 0) {
                currentTarget = scrambled ? 5 : 4;
              } else {
                currentTarget = scrambled ? 4 : 2;
              }
              break;
              case 4:
              if (scrmblHeading == 0) {
                currentTarget = scrambled ? 3 : 5;
              } else {
                currentTarget = scrambled ? 2 : 3;
              }
              break;
              case 5:
              if (scrmblHeading == 0) {
                currentTarget = 6;
                scrmblHeading = 180;
              } else {
                currentTarget = scrambled ? 3 : 4;
              }
              break;
              case 6:
              currentTarget = 5;
              break;
              default:
              break;
            }

            mouseScrambling.setHeading(scrmblHeading);
            mouseScrambling.setCenterLocation((int) ((currentTarget - 0.5)*scrmTargetWidth), (int) (worldHeight - tileHeight/2));
          }
          sim.iterate();
        }
      }

    });

  }

  void runAbridgingTrial() {
    System.out.println("Abridging trial running");
    stimInterrupt = false;
    resetAbrMouse();
    mouseAbridging.setVelocityX(motionSpeed);
    double rightEdge = (6.0 - deflectFactor)*abrReceptorWidth;
    double leftEdge = deflectFactor*abrReceptorWidth; // for better spatial locality during fast-paced calculations below
    motionSpeed = Integer.parseInt(abrSpeedField.getText());
    deflectFactor = abridged ? 0.35 + 0.12 * Math.floor(motionSpeed / 20) : 0.2 + 0.12 * Math.floor(motionSpeed / 20);
    System.out.print("Motion speed :");
    System.out.println(motionSpeed);
    System.out.print("Deflect factor: ");
    System.out.println(deflectFactor);
    Executors.newSingleThreadExecutor().execute(new Runnable() {
      @Override
      public void run() {

        workspace.run();

        while (!stimInterrupt) {
          if (abrHeading == 0 && mouseAbridging.getCenterX() > rightEdge) {
            abrHeading = 180;
            motionSpeed *= -1;
            mouseAbridging.setVelocityX(motionSpeed);
            mouseAbridging.setHeading(abrHeading);
          } else if (abrHeading == 180 && mouseAbridging.getCenterX() < leftEdge) {
            abrHeading = 0;
            motionSpeed *= -1;
            mouseAbridging.setVelocityX(motionSpeed);
            mouseAbridging.setHeading(abrHeading);
          } else if (abridged == true && abrHeading == 0 && mouseAbridging.getX() > 2.1*abrReceptorWidth && mouseAbridging.getX() < 2.5*abrReceptorWidth) {
            mouseAbridging.setX(4*abrReceptorWidth);
          } else if (abridged == true && abrHeading == 180 && mouseAbridging.getX() < 3.9*abrReceptorWidth && mouseAbridging.getX() > 3.5*abrReceptorWidth) {
            mouseAbridging.setX(2*abrReceptorWidth);
          }
        }
        System.out.println("Returning from abridging trial");
        return;
      }
    });
  }

  // void latchWait() throws AbortedException {
  //   latch.await();
  //   if (aborted) {
  //       throw new AbortedException();
  //       this.aborted = false;
  //   }
  // }

  // void abort() {
  //   this.aborted = true;
  //   latch.countDown();
  // }

  void setupListeners() {
    stimulusSelect.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent arg0) {
        String selectedStimulus = (String) stimulusSelect.getSelectedItem();
        if (selectedStimulus.equals("abridging")) {
          prevStim = currentStim;
          currentStim = Stimulus.ABR;
        } else if (selectedStimulus.equals("scrambling")) {
          prevStim = currentStim;
          currentStim = Stimulus.SCR;
        }
      }
    });

    freezeNet.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent arg0) {
        frozenWeights = !frozenWeights;
        String test = frozenWeights ? "frozen" : "plastic";
        SORNNetwork.freezeSynapses(frozenWeights);
        System.out.println("Weights should be " + test);
      }
    });

    homeostaticPlasticity.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent arg0) {
        homeoPlastic = !homeoPlastic;
        double eta_ip = homeoPlastic ? etaIP : 0;
        String test = homeoPlastic ? "on" : "off";
        str.setEtaIP(eta_ip);
        System.out.println("Plasticity should be " + test);
      }
    });

    occluderToggle.addActionListener(new ActionListener() {
      public void actionPerformed(ActionEvent arg0) {
        occluded = !occluded;
        String test = occluded ? "on" : "off";
        // Sensor_6 = target C tile (occluder); neuron_204 = corresponding neuron (remove this coupling)
        if (currentStim == Stimulus.ABR) {
          occluderCoupling(occluded);
        }
        System.out.println("Occluder is now turned " + test);
      }
    });

  }

  void occluderCoupling(boolean occlude) {
    try {
      if (occlude) {
        couple.removeMatchingCoupling(o1Coupling);
        couple.removeMatchingCoupling(o2Coupling);
      } else {
        couple.addCoupling(o1Coupling);
        couple.addCoupling(o2Coupling);
      }
    } catch (MismatchedAttributesException e) {
      e.printStackTrace();
    }
  }

  void abrCoupling() {

    couple.removeMatchingCoupling(t1Coupling);
    couple.removeMatchingCoupling(t2Coupling);
    couple.removeMatchingCoupling(t3Coupling);
    couple.removeMatchingCoupling(t4Coupling);
    couple.removeMatchingCoupling(t5Coupling);
    couple.removeMatchingCoupling(t6Coupling);

    try {
      couple.addCoupling(aCoupling);
      couple.addCoupling(bCoupling);
      couple.addCoupling(dCoupling);
      couple.addCoupling(eCoupling);
    } catch (MismatchedAttributesException e) {
      e.printStackTrace();
    }

    if (!occluded) {
      occluderCoupling(false);
    }
  }

  void scrCoupling() {
    couple.removeMatchingCoupling(aCoupling);
    couple.removeMatchingCoupling(bCoupling);
    couple.removeMatchingCoupling(dCoupling);
    couple.removeMatchingCoupling(eCoupling);
    if (!occluded) {
      occluderCoupling(true);
    }

    try {
      couple.addCoupling(t1Coupling);
      couple.addCoupling(t2Coupling);
      couple.addCoupling(t3Coupling);
      couple.addCoupling(t4Coupling);
      couple.addCoupling(t5Coupling);
      couple.addCoupling(t6Coupling);
    } catch (MismatchedAttributesException e) {
      e.printStackTrace();
    }
  }

  void makeControlPanel() {
    controlPanel = ControlPanel.makePanel(sim, "Control panel", -6, 1);
    controlPanel.addItem("Stimulus type", stimulusSelect);
    controlPanel.addItem("Freeze/unfreeze weights", freezeNet);
    controlPanel.addItem("Homeostatic plasticity", homeostaticPlasticity);
    controlPanel.addButton("Hide network (less lag)", "Hide", () -> {
      SORNvisible = !SORNvisible;
      SORNComponent.setGuiOn(SORNvisible);
      workspace.iterate();
    });

    netTimeStep = controlPanel.addTextField("Relative time step", "" + time_step);

    controlPanel.addSeparator();

    controlPanel.addItem("Toggle occluder", occluderToggle);
    controlPanel.addButton("Single/Double", "Abridge", () -> {
      stimInterrupt = true;
      workspace.stop();
      abridged = !abridged;
      runAbridgingTrial();
    });

    abrSpeedField = controlPanel.addTextField("Brushing speed", "" + motionSpeed);

    controlPanel.addSeparator();

    controlPanel.addButton("Ordered/scrambled", "Scramble", () -> {
      scrambled = !scrambled;
    });

    scrSOAField = controlPanel.addTextField("SOA (ms)", "" + scrmblSOA);

    controlPanel.addSeparator();

    controlPanel.addButton("Run", () -> {
      workspace.stop();
      // fixPlotColoring();
      SORNNetwork.setTimeStep(t_factor*time_step);
      stimulusNetwork.setTimeStep(t_factor*time_step);
      workspaceRunning = true;
      // Thread.currentThread().interrupt();
      stimInterrupt = true;
      if (currentStim == Stimulus.ABR) {
        resetScrmMouse();
        if (currentStim != prevStim) {
          abrCoupling();
        }
        prevStim = Stimulus.ABR;
        runAbridgingTrial();
      } else if (currentStim == Stimulus.SCR) {
        resetAbrMouse();
        if (currentStim != prevStim) {
          scrCoupling();
        }
        prevStim = Stimulus.SCR;
        runScramblingTrial();
      }
    });

    controlPanel.addButton("Stop", () -> {
      workspaceRunning = false;
      // Thread.currentThread().interrupt();
      stimInterrupt = true;
      resetAbrMouse();
      resetScrmMouse();
      workspace.stop();
    });

    controlPanel.addButton("Pause/step", () -> {
      if (workspaceRunning) {
        workspaceRunning = false;
        workspace.stop();
      } else {
        workspace.iterate();
      }
    });

    controlPanel.addButton("Reset network", () -> {
      SORNNetwork.clearActivations();
      SORNNetwork.clearInputs();
      sg_ee.randomizeConnectionWeights();
      sg_ie.randomizeConnectionWeights();
      sg_ei.randomizeConnectionWeights();
      SORNComponent.setGuiOn(true);
      SORNComponent.setGuiOn(SORNvisible);
    });

    // controlPanel.addButton("Fix plot colours", () -> {
    //   fixPlotColoring();
    // });
  }

  void setupGraphs() {
    plotPCAIn = sim.addProjectionPlot(285,410,615,430,"PCAIn");
    plotPCAEx = sim.addProjectionPlot(910,410,830,430,"PCAEx");
    exPCAComponent = plotPCAEx.getProjectionPlotComponent();
    inPCAComponent = plotPCAIn.getProjectionPlotComponent();
    sim.couple(SORNComponent, ng, exPCAComponent);
    sim.couple(SORNComponent, ngIn, inPCAComponent);
    plotPCAExProj = plotPCAEx.getProjectionModel().getProjector();
    plotPCAInProj = plotPCAIn.getProjectionModel().getProjector();
    plotPCAExClrMgr = plotPCAExProj.getColorManager();
    plotPCAInClrMgr = plotPCAInProj.getColorManager();
    fixPlotColoring();
    // plotPCAExClrMgr.closeDialogOk();
    // plotPCAInClrMgr.closeDialogOk();



  }

}
