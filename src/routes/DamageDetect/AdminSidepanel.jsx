import {useState} from "react";
import { Link } from "react-router-dom";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faBox, faUser, faBoxesPacking, faStreetView, faHeadset, faChainBroken, faHome } from '@fortawesome/free-solid-svg-icons'

function AdminSidepanel()
{

   

    return(

            <div className="w-[300px] side-panel p-5 flex flex-col items-center bg-[#d7002a] ">

                <div className="sidepalen-body h-2/3 p-3 w-full justify-center">
                    <div className="user-profile flex flex-row items-center space-x-5">
                        <label className="text-[28px]">
                            <FontAwesomeIcon icon={faUser} color="white" />
                        </label>
                        <span className="user-name text-[22px] text-white">User Name</span>
                    </div>


                    <div className="nav-body w-full flex-col justify-center space-y-5 mt-10">
                            <ul className="space-y-5">
                                <Link to="/Dashboard"><li className="p-2 w-full bg-white rounded-md mb-5">
                                    <FontAwesomeIcon icon={faHome} className="mr-5"/>Dashboard</li></Link>
                                <Link to="/AdminHome"><li className="p-2 w-full bg-white rounded-md mb-5">
                                    <FontAwesomeIcon icon={faBox} className="mr-5"/>Home</li></Link>

                                {/* <Link to="/PackageDispatch"><li className="p-2 w-full bg-slate-400 rounded-md mb-5">
                                    <FontAwesomeIcon icon={faBoxesPacking} className="mr-5"/>Package Dispatch</li></Link> */}

                                
                               
                            </ul>
                    </div>


                </div>
               

            </div>
            

    );
}


export default AdminSidepanel;