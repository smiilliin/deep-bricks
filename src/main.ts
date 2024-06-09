import * as PIXI from "pixi.js";
import { Vector2, getMapVectorFromScreen } from "./math";
import {
  Bricks,
  Conv2D,
  Dense,
  Flatten,
  IOutputConfig,
  ImageInput,
  Input,
  Output,
  Pooling,
} from "./bricks";
import { Nodes } from "./nodes";
import {
  ArgmaxModule,
  CoreModule,
  DrawingModule,
  Module,
  NumberSelectModule,
  PointInputModule,
  PointOutputModule,
  PointZInputModule,
  PointZOutputModule,
  Printmodule,
  RangeInputModule,
  TensorPackModule,
  TensorUnpackModule,
} from "./modules";
import * as tf from "@tensorflow/tfjs";
// import { ActivationIdentifier } from "@tensorflow/tfjs-layers/dist/keras_format/activation_config";
// import { InitializerIdentifier } from "@tensorflow/tfjs-layers/dist/initializers";

const app = new PIXI.Application({
  background: "#000000",
  resizeTo: document.body,
});

document.body.appendChild(app.view as HTMLCanvasElement);

app.stage.eventMode = "static";
app.stage.hitArea = app.screen;

const view = new PIXI.Container();

view.x = app.screen.width / 2;
view.y = app.screen.height / 2;

app.stage.addChild(view);

let viewMovedTimeout: NodeJS.Timeout | null = null;
function viewMoved() {
  if (!viewMovedTimeout) {
    viewMovedTimeout = setTimeout(() => {
      // world.viewMoved(view, f);
      viewMovedTimeout = null;
    }, 200);
  }
}

let mouseHolding = false;
app.stage.on("pointerdown", (event) => {
  if (event.button == 1) {
    mouseHolding = true;
    document.body.style.cursor = "grab";
  }
});
app.stage.on("pointermove", (event) => {
  if (mouseHolding) {
    view.x += event.movementX;
    view.y += event.movementY;
    viewMoved();
  }
});
const pointerLeave = () => {
  if (mouseHolding) {
    mouseHolding = false;

    document.body.style.cursor = "auto";
  }
};
app.stage.on("pointerup", pointerLeave);
app.stage.on("pointerleave", pointerLeave);
window.addEventListener("resize", () => {
  viewMoved();
});
app.stage.on("wheel", (event) => {
  const oldMouse = getMapVectorFromScreen(new Vector2(event.x, event.y), view);

  view.scale.x *= 1 - event.deltaY / 1000;
  view.scale.y *= 1 - event.deltaY / 1000;

  if (view.scale.x < 0.3) {
    view.scale.x = 0.3;
    view.scale.y = 0.3;
    return;
  }
  if (view.scale.x > 2) {
    view.scale.x = 2;
    view.scale.y = 2;
  }

  const newMouse = getMapVectorFromScreen(new Vector2(event.x, event.y), view);

  const movement = newMouse.sub(oldMouse).mul(view.scale.x);

  view.x += movement.x;
  view.y += movement.y;

  viewMoved();
});

app.ticker.add(() => {});

const bricks = new Bricks();

// view.addChild(bricks);

view.sortableChildren = true;

function addNewBricksButtonListener() {
  const newInput = document.getElementById("newinput") as HTMLSpanElement;
  const newImage = document.getElementById("newimage") as HTMLSpanElement;
  const newConv = document.getElementById("newconv") as HTMLSpanElement;
  const newPool = document.getElementById("newpool") as HTMLSpanElement;
  const newDense = document.getElementById("newdense") as HTMLSpanElement;
  const newFlatten = document.getElementById("newflatten") as HTMLSpanElement;
  const newOutput = document.getElementById("newoutput") as HTMLSpanElement;
  const done = document.getElementById("done") as HTMLSpanElement;

  newInput.onclick = () => {
    const brick = new Input(bricks);
    brick.registerEvents(app, view, bricks);
    bricks.addBrick(brick);
    bricks.sort();
  };
  newImage.onclick = () => {
    const brick = new ImageInput(bricks);
    brick.registerEvents(app, view, bricks);
    bricks.addBrick(brick);
    bricks.sort();
  };
  newConv.onclick = () => {
    const brick = new Conv2D(bricks);
    brick.registerEvents(app, view, bricks);
    bricks.addBrick(brick);
    bricks.sort();
  };
  newPool.onclick = () => {
    const brick = new Pooling(bricks);
    brick.registerEvents(app, view, bricks);
    bricks.addBrick(brick);
    bricks.sort();
  };
  newDense.onclick = () => {
    const brick = new Dense(bricks);
    brick.registerEvents(app, view, bricks);
    bricks.addBrick(brick);
    bricks.sort();
  };
  newFlatten.onclick = () => {
    const brick = new Flatten(bricks);
    brick.registerEvents(app, view, bricks);
    bricks.addBrick(brick);
    bricks.sort();
  };
  newOutput.onclick = () => {
    const brick = new Output(bricks);
    brick.registerEvents(app, view, bricks);
    bricks.addBrick(brick);
    bricks.sort();
  };
  done.onclick = () => {
    try {
      const model: tf.Sequential = tf.sequential();
      let compileOption: IOutputConfig | null = null;
      let inputShape: number[] = [];
      let outputShape: number = 0;

      bricks.bricks.forEach((brick) => {
        const layer = brick.toLayer();

        if (!layer) return;

        if (brick instanceof Input || brick instanceof ImageInput) {
          inputShape = brick.inputShape;
        }

        if (brick instanceof Dense) {
          outputShape = brick.units;
        }

        if (layer instanceof tf.layers.Layer) {
          model.add(layer);
        } else {
          compileOption = layer;
        }
      });
      if (compileOption != null && inputShape.length != 0 && outputShape != 0) {
        model.compile((compileOption as IOutputConfig).config);
        coreModule.model = model;
        coreModule.inputShape = inputShape;
        coreModule.outputShape = outputShape;
        coreModule.epochs = (compileOption as IOutputConfig).epochs;

        view.removeChild(bricks);
        view.addChild(nodes);
        newBricks.hidden = true;
        bricksets.hidden = true;
        modules.forEach((m) => view.addChild(m));
      }
    } catch (err) {
      if (err instanceof Error) {
        alert(err.message);
      }
    }
  };
}
function addBricksetsButtonListener() {
  const resetBricks = () => {
    bricks.bricks.forEach((brick) => {
      bricks.removeChild(brick);
    });
    bricks.bricks = [];
  };

  const set1D = document.getElementById("set1d") as HTMLSpanElement;
  const set2D = document.getElementById("set2d") as HTMLSpanElement;
  const setCNN = document.getElementById("setcnn") as HTMLSpanElement;

  set1D.onclick = () => {
    resetBricks();
    const input = new Input(bricks);

    input.units = 64;
    input.inputShape = [1];
    input.activation = "relu";

    const dense = new Dense(bricks);

    dense.units = 32;
    dense.activation = "relu";

    const dense2 = new Dense(bricks);

    dense2.units = 1;
    dense2.activation = undefined;

    const output = new Output(bricks);

    output.loss = "meanSquaredError";
    output.epochs = 2000;

    input.registerEvents(app, view, bricks);
    dense.registerEvents(app, view, bricks);
    dense2.registerEvents(app, view, bricks);
    output.registerEvents(app, view, bricks);
    bricks.addBrick(input);
    bricks.addBrick(dense);
    bricks.addBrick(dense2);
    bricks.addBrick(output);
    bricks.sort();
  };
  set2D.onclick = () => {
    resetBricks();
    const input = new Input(bricks);

    input.units = 64;
    input.inputShape = [2];
    input.activation = "relu";

    const dense = new Dense(bricks);

    dense.units = 32;
    dense.activation = "relu";

    const dense2 = new Dense(bricks);

    dense2.units = 1;
    dense2.activation = undefined;

    const output = new Output(bricks);

    output.loss = "meanSquaredError";
    output.epochs = 2000;

    input.registerEvents(app, view, bricks);
    dense.registerEvents(app, view, bricks);
    dense2.registerEvents(app, view, bricks);
    output.registerEvents(app, view, bricks);
    bricks.addBrick(input);
    bricks.addBrick(dense);
    bricks.addBrick(dense2);
    bricks.addBrick(output);
    bricks.sort();
  };
  setCNN.onclick = () => {
    resetBricks();
    const input = new ImageInput(bricks);

    input.inputShape = [128, 128, 1];
    input.kernelSize = 5;
    input._filters = 8;
    input.strides = 1;
    input.activation = "relu";
    input.kernelInitializer = "varianceScaling";

    const pooling = new Pooling(bricks);

    pooling.poolSize = [2, 2];
    pooling.strides = [2, 2];

    const conv = new Conv2D(bricks);
    conv.kernelSize = 5;
    conv._filters = 16;
    conv.strides = 1;
    conv.activation = "relu";
    conv.kernelInitializer = "varianceScaling";

    const pooling2 = new Pooling(bricks);

    pooling2.poolSize = [2, 2];
    pooling2.strides = [2, 2];

    const flatten = new Flatten(bricks);

    const dense = new Dense(bricks);

    dense.units = 10;
    dense.kernelInitializer = "varianceScaling";
    dense.activation = "softmax";

    const output = new Output(bricks);

    output.loss = "categoricalCrossentropy";
    output.epochs = 300;

    input.registerEvents(app, view, bricks);
    pooling.registerEvents(app, view, bricks);
    conv.registerEvents(app, view, bricks);
    pooling2.registerEvents(app, view, bricks);
    flatten.registerEvents(app, view, bricks);
    dense.registerEvents(app, view, bricks);
    output.registerEvents(app, view, bricks);

    bricks.addBrick(input);
    bricks.addBrick(pooling);
    bricks.addBrick(conv);
    bricks.addBrick(pooling2);
    bricks.addBrick(flatten);
    bricks.addBrick(dense);
    bricks.addBrick(output);
    bricks.sort();
  };
}

const newBricks = document.getElementById("newBricks") as HTMLDivElement;
const bricksets = document.getElementById("bricksets") as HTMLDivElement;

addNewBricksButtonListener();
addBricksetsButtonListener();

const nodes = new Nodes(app);

view.addChild(nodes);
nodes.zIndex = Number.MAX_VALUE;

const modules: Module[] = [];

const makeNewModule = (m: Module, x: number, y: number) => {
  m.x = x;
  m.y = y;
  modules.push(m);
  m.update(nodes);
  m.registerEvents(app, view, nodes);
  view.addChild(m);
};

const coreModule = new CoreModule(nodes);

coreModule.onModuleEditOpen = () => {
  modules.forEach((m) => view.removeChild(m));

  newBricks.style.display = "flex";
  bricksets.style.display = "flex";

  view.removeChild(nodes);
  view.addChild(bricks);
};

makeNewModule(coreModule, 0, 0);
makeNewModule(new PointInputModule(nodes, view), -600, -100);
makeNewModule(new PointOutputModule(nodes), 500, -100);
makeNewModule(new PointZInputModule(nodes, view), -600, 600);
makeNewModule(new PointZOutputModule(nodes), 500, 600);
makeNewModule(new RangeInputModule(nodes, 2), 0, -400);
makeNewModule(new RangeInputModule(nodes, 1), 0, -600);
makeNewModule(new TensorPackModule(nodes, 1), 500, -400);
makeNewModule(new TensorPackModule(nodes, 1), 500, -800);
makeNewModule(new TensorUnpackModule(nodes, 1), 700, -400);
makeNewModule(new TensorUnpackModule(nodes, 1), 700, -800);
makeNewModule(new TensorPackModule(nodes, 2), 1000, -400);
makeNewModule(new TensorPackModule(nodes, 2), 1000, -800);
makeNewModule(new TensorUnpackModule(nodes, 2), 1200, -400);
makeNewModule(new TensorUnpackModule(nodes, 2), 1200, -800);
makeNewModule(new DrawingModule(nodes, view), -100, -1200);
makeNewModule(new NumberSelectModule(nodes), -500, -800);
makeNewModule(new ArgmaxModule(nodes), -800, -800);
makeNewModule(new Printmodule(nodes), -800, -500);

// const coreModule = new CoreModule(nodes, 300);
// const pointInputModule = new PointInputModule(nodes, view);
// const pointOutputModule = new PointOutputModule(nodes);
// const pointZInputModule = new PointZInputModule(nodes, view);
// const pointZOutputModule = new PointZOutputModule(nodes);
// const rangeInputModule = new RangeInputModule(nodes, 2);
// const rangeInputModule2 = new RangeInputModule(nodes, 1);
// const tensorPackModule = new TensorPackModule(nodes, 1);
// const tensorPackModule2 = new TensorPackModule(nodes, 1);
// const tensorUnpackModule = new TensorUnpackModule(nodes, 1);
// const tensorUnpackModule2 = new TensorUnpackModule(nodes, 1);
// const tensorPackModule3 = new TensorPackModule(nodes, 2);
// const tensorPackModule4 = new TensorPackModule(nodes, 2);
// const tensorUnpackModule3 = new TensorUnpackModule(nodes, 2);
// const tensorUnpackModule4 = new TensorUnpackModule(nodes, 2);
// const drawingModule = new DrawingModule(nodes, view);
// const numberSelectModule = new NumberSelectModule(nodes);
// const argmaxModule = new ArgmaxModule(nodes);
// const printModule = new Printmodule(nodes);

// pointInputModule.x = -600;
// pointInputModule.y = -100;

// pointOutputModule.x = 500;
// pointOutputModule.y = -100;

// pointZInputModule.x = -600;
// pointZInputModule.y = 600;

// pointZOutputModule.x = 500;
// pointZOutputModule.y = 600;

// rangeInputModule.x = 0;
// rangeInputModule.y = -400;

// rangeInputModule2.x = 0;
// rangeInputModule2.y = -600;

// tensorPackModule.x = 500;
// tensorPackModule.y = -400;

// tensorUnpackModule.x = 500;
// tensorUnpackModule.y = -800;

// tensorPackModule2.x = 700;
// tensorPackModule2.y = -400;

// tensorUnpackModule2.x = 700;
// tensorUnpackModule2.y = -800;

// tensorPackModule3.x = 1000;
// tensorPackModule3.y = -400;

// tensorUnpackModule3.x = 1000;
// tensorUnpackModule3.y = -800;

// tensorPackModule4.x = 1200;
// tensorPackModule4.y = -400;

// tensorUnpackModule4.x = 1200;
// tensorUnpackModule4.y = -800;

// drawingModule.x = -100;
// drawingModule.y = -1200;

// numberSelectModule.x = -500;
// numberSelectModule.y = -800;

// argmaxModule.x = -800;
// argmaxModule.y = -800;

// printModule.x = -800;
// printModule.y = -500;

// coreModule.update(nodes);
// pointInputModule.update(nodes);
// pointOutputModule.update(nodes);
// pointZInputModule.update(nodes);
// pointZOutputModule.update(nodes);
// rangeInputModule.update(nodes);
// rangeInputModule2.update(nodes);
// tensorPackModule.update(nodes);
// tensorPackModule2.update(nodes);
// tensorPackModule3.update(nodes);
// tensorPackModule4.update(nodes);
// tensorUnpackModule.update(nodes);
// tensorUnpackModule2.update(nodes);
// tensorUnpackModule3.update(nodes);
// tensorUnpackModule4.update(nodes);
// drawingModule.update(nodes);
// numberSelectModule.update(nodes);
// argmaxModule.update(nodes);
// printModule.update(nodes);

// coreModule.registerEvents(app, view, nodes);
// pointInputModule.registerEvents(app, view, nodes);
// pointOutputModule.registerEvents(app, view, nodes);
// pointZInputModule.registerEvents(app, view, nodes);
// pointZOutputModule.registerEvents(app, view, nodes);
// rangeInputModule.registerEvents(app, view, nodes);
// rangeInputModule2.registerEvents(app, view, nodes);
// tensorPackModule.registerEvents(app, view, nodes);
// tensorPackModule2.registerEvents(app, view, nodes);
// tensorPackModule3.registerEvents(app, view, nodes);
// tensorPackModule4.registerEvents(app, view, nodes);
// tensorUnpackModule.registerEvents(app, view, nodes);
// tensorUnpackModule2.registerEvents(app, view, nodes);
// tensorUnpackModule3.registerEvents(app, view, nodes);
// tensorUnpackModule4.registerEvents(app, view, nodes);
// drawingModule.registerEvents(app, view, nodes);
// numberSelectModule.registerEvents(app, view, nodes);
// argmaxModule.registerEvents(app, view, nodes);
// printModule.registerEvents(app, view, nodes);

// view.addChild(coreModule);
// view.addChild(pointInputModule);
// view.addChild(pointOutputModule);
// view.addChild(pointZInputModule);
// view.addChild(pointZOutputModule);
// view.addChild(rangeInputModule);
// view.addChild(rangeInputModule2);
// view.addChild(tensorPackModule);
// view.addChild(tensorPackModule2);
// view.addChild(tensorPackModule3);
// view.addChild(tensorPackModule4);
// view.addChild(tensorUnpackModule);
// view.addChild(tensorUnpackModule2);
// view.addChild(tensorUnpackModule3);
// view.addChild(tensorUnpackModule4);
// view.addChild(drawingModule);
// view.addChild(numberSelectModule);
// view.addChild(argmaxModule);
// view.addChild(printModule);

// coreModule.inputModule.onDataPassed = (data) => {
//   console.log(data);
// };

document.body.oncontextmenu = (e) => {
  e.preventDefault();
};
