import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import homeLogo from "../../Assets/danny_picture.png";
import Particle from "../Particle";
// import Type from "./Type";
import Tilt from "react-parallax-tilt";

function Home() {
  return (
    <section>
      <Container fluid className="home-section" id="home">
        <Particle />
        <Container className="home-content">
        <Row>
            <Col md={7} className="home-header">
            <h1 style={{ paddingBottom: 15 }} className="heading">
              G'Day, Mate!{" "}
              <span className="kangaroo">🦘</span>
            </h1>

              <h1 className="heading-name">
                I'm
                <strong className="main-name"> Danny Nguyen</strong>
              </h1>
            </Col>

            <Col md={5} style={{ paddingBottom: 20 }}>
              <Tilt>
                <img
                  src={homeLogo}
                  alt="home pic"
                  className="img-fluid"
                  style={{ maxHeight: "450px" }}
                />
              </Tilt>
            </Col>
          </Row>
        </Container>
      </Container>
    </section>
  );
}

export default Home;
