import * as PIXI from "pixi.js";
import { Vector2 } from "./math";

class NodeModule extends PIXI.Container {
  next: NodeModule | null;
  getGlobalVector: () => Vector2;
  onDataPassed: (data: unknown, passCount: number) => void;
  circle: PIXI.Graphics;
  text: PIXI.Text;

  constructor(nodes: Nodes, text?: string) {
    super();
    this.circle = new PIXI.Graphics();
    this.text = new PIXI.Text(text, {
      fill: 0xffffff,
      align: "center",
      fontSize: 18,
    });
    this.text.anchor.set(0.5, 0);
    this.text.x = 0;
    this.text.y = 15;

    this.addChild(this.circle);
    this.addChild(this.text);
    this.next = null;

    this.unselected();
    this.getGlobalVector = () => new Vector2(this.x, this.y);
    this.eventMode = "static";

    nodes.add(this);
    this.makeInteractive(nodes);

    this.onDataPassed = () => {};
  }
  setGlobalVector(containerVector: Vector2) {
    const globalVector = containerVector.add(new Vector2(this.x, this.y));
    this.getGlobalVector = () => globalVector;
  }
  disconnect() {
    if (!this.next) return;
    this.next.next = null;
    this.next = null;
  }
  connect(n: NodeModule) {
    this.next = n;
    n.next = this;
  }
  selected() {
    this.circle.clear();
    this.circle.beginFill(0xff1111);
    this.circle.drawCircle(0, 0, 15);
    this.circle.endFill();
  }
  unselected() {
    this.circle.clear();
    this.circle.beginFill(0xffffff);
    this.circle.drawCircle(0, 0, 15);
    this.circle.endFill();
  }
  makeInteractive(nodes: Nodes) {
    this.on("pointerdown", (event) => {
      event.stopPropagation();

      if (event.button == 0) {
        if (Nodes.currentNode && this != Nodes.currentNode) {
          nodes.connect(this, Nodes.currentNode);
          nodes.update();
          Nodes.currentNode.unselected();
          Nodes.currentNode = null;
        } else {
          Nodes.currentNode = this;
          this.selected();
        }
      } else if (event.button == 2) {
        if (this.next) {
          this.next.next = null;
          this.next = null;
          nodes.update();
        }
      }
    });
  }
  passData(data: unknown, passCount?: number) {
    if ((passCount || 0) > 100) {
      console.warn("Excessed count of passing data detected");
      return;
    }

    this.onDataPassed(data, passCount || 0);
    this.next?.onDataPassed(data, (passCount || 0) + 1);
  }
}

class Nodes extends PIXI.Container {
  nodes: NodeModule[];
  connected: Map<number, boolean>;
  lines: PIXI.Graphics[];
  static currentNode: NodeModule | null;

  constructor(app: PIXI.Application) {
    super();
    this.nodes = [];
    this.connected = new Map();
    this.lines = [];

    app.stage.on("pointerdown", () => {
      Nodes.currentNode?.unselected();
      Nodes.currentNode = null;
    });
  }

  update() {
    this.lines.forEach((line) => this.removeChild(line));
    this.lines.splice(0);

    this.connected.forEach((b, i) => {
      if (!b) return;
      const n = this.nodes[i];

      if (n.next == null) {
        this.connected.delete(i);
      } else {
        const line = new PIXI.Graphics();

        line.clear();
        line.lineStyle(4, 0xffffff);

        const nVector = n.getGlobalVector();
        const nextVector = n.next.getGlobalVector();
        line.moveTo(nVector.x, nVector.y);
        line.lineTo(nextVector.x, nextVector.y);

        this.addChild(line);
        this.lines.push(line);
      }
    });
  }
  connect(target: NodeModule, target2: NodeModule) {
    const i = this.nodes.indexOf(target);
    const i2 = this.nodes.indexOf(target2);
    if (i == -1 || i2 == -1) return;

    const nextI = target.next != null ? this.nodes.indexOf(target.next) : null;
    const nextI2 =
      target2.next != null ? this.nodes.indexOf(target2.next) : null;

    if (nextI != null) {
      target.next!.next = null;
      target.next = null;
      this.connected.delete(Math.min(i, nextI));
    }
    if (nextI2 != null) {
      target2.next!.next = null;
      target2.next = null;
      this.connected.delete(Math.min(i2, nextI2));
    }

    target.connect(target2);
    const minI = Math.min(i, i2);

    this.connected.set(minI, true);
  }
  add(n: NodeModule) {
    this.nodes.push(n);
  }
  get(i: number) {
    return this.nodes[i];
  }
}

export { NodeModule, Nodes };
