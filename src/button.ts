import * as PIXI from "pixi.js";
import { Vector2 } from "./math";

class Button extends PIXI.Container {
  body: PIXI.Graphics;
  text: PIXI.Text;
  size: Vector2;

  constructor(
    size: Vector2,
    color: PIXI.ColorSource,
    textcolor: PIXI.TextStyleFill,
    text: string,
    fontsize: number
  ) {
    super();
    this.size = size;

    this.body = new PIXI.Graphics();
    this.body.beginFill(color);
    this.body.drawRect(0, 0, size.x, size.y);
    this.body.endFill();

    this.text = new PIXI.Text(text, {
      fill: textcolor,
      align: "center",
      fontSize: fontsize,
    });
    this.text.anchor.set(0.5, 0.5);
    this.text.x = size.x / 2;
    this.text.y = size.y / 2;

    this.eventMode = "static";
    this.on("pointerdown", (event) => event.stopPropagation());

    this.addChild(this.body);
    this.addChild(this.text);
  }
}
class SelectButton extends Button {
  selected: boolean;
  textcolor: PIXI.TextStyleFill;
  selectcolor: PIXI.TextStyleFill;

  constructor(
    size: Vector2,
    color: PIXI.ColorSource,
    textcolor: PIXI.TextStyleFill,
    selectcolor: PIXI.TextStyleFill,
    text: string,
    fontsize: number
  ) {
    super(size, color, textcolor, text, fontsize);
    this.selected = false;

    this.on("pointerdown", (event) => event.stopPropagation());
    this.on("click", () => {
      this.selected = !this.selected;
      this.update();
    });
    this.textcolor = textcolor;
    this.selectcolor = selectcolor;
  }
  unselect() {
    this.selected = false;
    this.update();
  }
  select() {
    this.selected = true;
    this.update();
  }
  update() {
    if (this.selected) {
      this.text.style.fill = this.selectcolor;
    } else {
      this.text.style.fill = this.textcolor;
    }
  }
}

export { Button, SelectButton };
